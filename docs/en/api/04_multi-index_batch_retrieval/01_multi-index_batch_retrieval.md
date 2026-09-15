# Multi-Index Batch Search<a name="en-us_TOPIC_0000001456535132"></a>

When the `distances` values are the same, multi-Index batch search uses a different TopK ranking algorithm than single-Index search. Therefore, the labels in the returned TopK results can differ.

## `Search` (`AscendIndex`)<a name="en-us_TOPIC_0000001456854904"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13466158111313"><a name="p13466158111313"></a><a name="p13466158111313"></a><code>void Search(std::vector&lt;AscendIndex *&gt; indexes, idx_t n, const float *x, idx_t k,float *distances, idx_t *labels, bool merged);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p541145901013"><a name="p541145901013"></a><a name="p541145901013"></a>Provides an interface for searching feature vectors across multiple <code>AscendIndex</code> instances. It returns the distances and IDs of the <code>k</code> most similar features based on the input feature vectors.</p>
<p id="p92091230151318"><a name="p92091230151318"></a><a name="p92091230151318"></a>The following algorithms are currently supported:</p>
<a name="ul166141532131310"></a><a name="ul166141532131310"></a><ul id="ul166141532131310"><li>Subtypes derived from <code>Index</code>: <code>AscendIndexSQ</code> (<code>QuantizerType</code> = <code>QT_8bit</code>).</li><li>Subtypes derived from <code>Index</code>: <code>AscendIndexFlat</code> (<code>FlatIP</code>, <code>FlatL2</code>).</li><li>Subtypes derived from <code>Index</code>: <code>AscendIndexIVFSP</code>.</li></ul>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p98111229152410"><a name="p98111229152410"></a><a name="p98111229152410"></a><code>std::vector&lt;AscendIndex *&gt; indexes</code>: Indexes to be used in the search.</p>
<p id="p311738102412"><a name="p311738102412"></a><a name="p311738102412"></a><code>idx_t n</code>: Number of queries to be used in the search.</p>
<p id="p41091246162416"><a name="p41091246162416"></a><a name="p41091246162416"></a><code>const float *x</code>: Query feature vectors to be used in the search.</p>
<p id="p17381952152416"><a name="p17381952152416"></a><a name="p17381952152416"></a><code>idx_t k</code>: Number of most similar results to return.</p>
<p id="p11990043102211"><a name="p11990043102211"></a><a name="p11990043102211"></a><code>bool merged</code>: Whether to merge the results from search across multiple indexes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><code>float *distances</code>: Distance values between the query vectors and the <code>k</code> nearest vectors.</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><code>idx_t *labels</code>: IDs of the <code>k</code> nearest vectors to the queries.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul625373719595"></a><a name="ul625373719595"></a><ul id="ul625373719595"><li>The indexes participating in search must be created on the same card.</li><li>The supported <code>indexes</code> types are as follows.<a name="ul8694125212459"></a><a name="ul8694125212459"></a><ul id="ul8694125212459"><li><code>indexes</code> points to <code>AscendIndexSQ</code>, and <code>QuantizerType</code> must be <code>QT_8bit</code>, with 0 &lt; <code>indexes.size()</code> ≤ 10000.</li><li><code>indexes</code> points to <code>AscendIndexIVFSP</code>, and the corresponding <code>QuantizerType</code> must be <code>QT_8bit</code> and <code>MetricType</code> must be <code>METRIC_L2</code>, with 0 &lt; <code>indexes.size()</code> ≤ 10000. The <code>AscendIndexIVFSP</code> indexes participating in search must share the same codebook in shared memory. You can create instances with the shared-codebook constructor provided by <code>AscendIndexIVFSP</code> or through the <code>loadAllData</code> interface.</li><li><code>indexes</code> points to <code>AscendIndexFlat</code>, with 0 &lt; <code>indexes.size()</code> ≤ 10000.</li></ul>
</li><li><code>n</code> must not exceed 1024.</li><li><code>k</code> must not exceed 1024.</li><li><code>x</code> must be a non-null pointer, and its length must be <code>dim * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>distances</code> and <code>labels</code> must be non-null pointers and must satisfy the following:<a name="ul136865113167"></a><a name="ul136865113167"></a><ul id="ul136865113167"><li>When <code>merged = true</code>, their length must be <code>k * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li>When <code>merged = false</code>, their length must be <code>indexes.size() * k * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

## `Search` (`AscendIndexInt8`)<a name="en-us_TOPIC_0000001533044201"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p62099237551"><a name="p62099237551"></a><a name="p62099237551"></a><code>void Search(std::vector&lt;AscendIndexInt8 *&gt; indexes, idx_t n, const int8_t *x, idx_t k, float *distances, idx_t *labels, bool merged);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1558519285114"><a name="p1558519285114"></a><a name="p1558519285114"></a>Provides an interface for searching feature vectors across multiple <code>AscendIndexInt8</code> instances. It returns the distances and IDs of the <code>k</code> most similar features based on the input feature vectors.</p>
<p id="p17585112881112"><a name="p17585112881112"></a><a name="p17585112881112"></a>Currently, only <code>AscendIndexInt8Flat</code>, a subtype derived from <code>AscendIndexInt8</code>, is supported.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p98111229152410"><a name="p98111229152410"></a><a name="p98111229152410"></a><code>std::vector&lt;AscendIndexInt8 *&gt; indexes</code>: Indexes to be used in the search.</p>
<p id="p311738102412"><a name="p311738102412"></a><a name="p311738102412"></a><code>idx_t n</code>: Number of queries to be used in the search.</p>
<p id="p41091246162416"><a name="p41091246162416"></a><a name="p41091246162416"></a><code>const int8_t *x</code>: Query feature vectors to be used in the search.</p>
<p id="p17381952152416"><a name="p17381952152416"></a><a name="p17381952152416"></a><code>idx_t k</code>: Number of most similar results to return.</p>
<p id="p11990043102211"><a name="p11990043102211"></a><a name="p11990043102211"></a><code>bool merged</code>: Whether to merge the results from search across multiple indexes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><code>float *distances</code>: Distance values between the query vectors and the <code>k</code> nearest vectors.</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><code>idx_t *labels</code>: IDs of the <code>k</code> nearest vectors to the queries.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul4866551111619"></a><a name="ul4866551111619"></a><ul id="ul4866551111619"><li>The indexes participating in search must be created on the same card.</li></ul>
<a name="ul625373719595"></a><a name="ul625373719595"></a><ul id="ul625373719595"><li>Only <code>AscendIndexInt8</code> is supported for <code>indexes</code>, with 0 &lt; <code>indexes.size()</code> ≤ 10000.</li><li><code>n</code> must not exceed 1024.</li><li><code>k</code> must not exceed 1024.</li><li><code>x</code> must be a non-null pointer, and its length must be <code>dim * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>distances</code> and <code>labels</code> must be non-null pointers and must satisfy the following:<a name="ul136865113167"></a><a name="ul136865113167"></a><ul id="ul136865113167"><li>When <code>merged = true</code>, their length must be <code>k * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li>When <code>merged = false</code>, their length must be <code>indexes.size() * k * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

## `Search` (`FaissIndex`)<a name="en-us_TOPIC_0000001506334841"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13466158111313"><a name="p13466158111313"></a><a name="p13466158111313"></a><code>void Search(std::vector&lt;Index *&gt; indexes, idx_t n, const float *x, idx_t k,float *distances, idx_t *labels, bool merged);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p412313151969"><a name="p412313151969"></a><a name="p412313151969"></a>Provides an interface for searching feature vectors across multiple <code>Index</code> instances. It returns the distances and IDs of the <code>k</code> most similar features based on the input feature vectors.</p>
<p id="p92091230151318"><a name="p92091230151318"></a><a name="p92091230151318"></a>The following algorithms are currently supported:</p>
<a name="ul166141532131310"></a><a name="ul166141532131310"></a><ul id="ul166141532131310"><li>Subtypes derived from <code>Index</code>: <code>AscendIndexSQ</code> (<code>QuantizerType</code> = <code>QT_8bit</code>).</li><li>Subtypes derived from <code>Index</code>: <code>AscendIndexFlat</code> (<code>FlatIP</code>, <code>FlatL2</code>).</li><li>Subtypes derived from <code>Index</code>: <code>AscendIndexIVFSP</code>.</li></ul>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p98111229152410"><a name="p98111229152410"></a><a name="p98111229152410"></a><code>std::vector&lt;Index *&gt; indexes</code>: Indexes to be used in the search.</p>
<p id="p311738102412"><a name="p311738102412"></a><a name="p311738102412"></a><code>idx_t n</code>: Number of queries to be used in the search.</p>
<p id="p41091246162416"><a name="p41091246162416"></a><a name="p41091246162416"></a><code>const float *x</code>: Query feature vectors to be used in the search.</p>
<p id="p17381952152416"><a name="p17381952152416"></a><a name="p17381952152416"></a><code>idx_t k</code>: Number of most similar results to return.</p>
<p id="p11990043102211"><a name="p11990043102211"></a><a name="p11990043102211"></a><code>bool merged</code>: Whether to merge the results from search across multiple indexes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><code>float *distances</code>: Distance values between the query vectors and the <code>k</code> nearest vectors.</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><code>idx_t *labels</code>: IDs of the <code>k</code> nearest vectors to the queries.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1876412513567"></a><a name="ul1876412513567"></a><ul id="ul1876412513567"><li>The indexes participating in search must be created on the same card.</li><li>The supported <code>indexes</code> types are as follows.<a name="ul8694125212459"></a><a name="ul8694125212459"></a><ul id="ul8694125212459"><li><code>indexes</code> points to <code>AscendIndexSQ</code>, and <code>QuantizerType</code> must be <code>QT_8bit</code>, with 0 &lt; <code>indexes.size()</code> ≤ 10000.</li><li><code>indexes</code> points to <code>AscendIndexIVFSP</code>, and the corresponding <code>QuantizerType</code> must be <code>QT_8bit</code> and <code>MetricType</code> must be <code>METRIC_L2</code>, with 0 &lt; <code>indexes.size()</code> ≤ 10000. The <code>AscendIndexIVFSP</code> indexes participating in search must share the same codebook in shared memory. You can create instances with the shared-codebook constructor provided by <code>AscendIndexIVFSP</code> or through the <code>loadAllData</code> interface.</li><li><code>indexes</code> points to <code>AscendIndexFlat</code>, with 0 &lt; <code>indexes.size()</code> ≤ 10000.</li></ul>
</li><li><code>n</code> must not exceed 1024.</li><li><code>k</code> must not exceed 1024.</li><li><code>x</code> must be a non-null pointer, and its length must be <code>dim * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>distances</code> and <code>labels</code> must be non-null pointers and must satisfy the following:<a name="ul136865113167"></a><a name="ul136865113167"></a><ul id="ul136865113167"><li>When <code>merged = true</code>, their length must be <code>k * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li>When <code>merged = false</code>, their length must be <code>indexes.size() * k * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

## `SearchWithFilter` (`FaissIndex`, Single Filter)<a name="en-us_TOPIC_0000001521615937"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2063619251347"><a name="p2063619251347"></a><a name="p2063619251347"></a><code>void SearchWithFilter(std::vector&lt;Index *&gt; indexes, idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters, bool merged);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Searches multiple <code>indexes</code> and returns the IDs of the <code>k</code> most similar features based on the input feature vectors. It supports CID-based filtering. <code>filters</code> is a <code>uint32_t</code> array of length <code>n * 6</code>, and every six <code>uint32_t</code> values form one filter. The first four numbers of each filter (128 bits) represent the corresponding CID, and the last two numbers represent the left-closed timestamp range, that is, [<code>x</code>, <code>y</code>).</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p98111229152410"><a name="p98111229152410"></a><a name="p98111229152410"></a><code>std::vector&lt;Index *&gt; indexes</code>: Indexes to be used in the search.</p>
<p id="p311738102412"><a name="p311738102412"></a><a name="p311738102412"></a><code>idx_t n</code>: Number of queries to be used in the search.</p>
<p id="p41091246162416"><a name="p41091246162416"></a><a name="p41091246162416"></a><code>const float *x</code>: Query feature vectors to be used in the search.</p>
<p id="p17381952152416"><a name="p17381952152416"></a><a name="p17381952152416"></a><code>idx_t k</code>: Number of most similar results to return.</p>
<p id="p425111215144"><a name="p425111215144"></a><a name="p425111215144"></a><code>const void *filters</code>: Filter conditions.</p>
<p id="p11990043102211"><a name="p11990043102211"></a><a name="p11990043102211"></a><code>bool merged</code>: Whether to merge the results from search across multiple indexes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><code>float *distances</code>: Distance values between the query vectors and the <code>k</code> nearest vectors.</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><code>idx_t *labels</code>: IDs of the <code>k</code> nearest vectors to the queries.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul151271598168"></a><a name="ul151271598168"></a><ul id="ul151271598168"><li>The indexes participating in search must be created on the same card.</li></ul>
<a name="ul625373719595"></a><a name="ul625373719595"></a><ul id="ul625373719595"><li>The supported <code>indexes</code> types are as follows.<a name="ul153841243520"></a><a name="ul153841243520"></a><ul id="ul153841243520"><li><code>indexes</code> points to <code>AscendIndexSQ</code>, and <code>QuantizerType</code> must be <code>QT_8bit</code>, with 0 &lt; <code>indexes.size()</code> ≤ 10000.</li><li><code>indexes</code> points to <code>AscendIndexIVFSP</code>, and the corresponding <code>QuantizerType</code> must be <code>QT_8bit</code> and <code>MetricType</code> must be <code>METRIC_L2</code>, with 0 &lt; <code>indexes.size()</code> ≤ 10000. The <code>AscendIndexIVFSP</code> indexes participating in search must share the same codebook in shared memory. You can create instances with the shared-codebook constructor provided by <code>AscendIndexIVFSP</code> or through the <code>loadAllData</code> interface.</li></ul>
</li><li><code>n</code> must not exceed 1024.</li><li><code>k</code> must not exceed 1024.</li><li><code>x</code> must be a non-null pointer, and its length must be <code>dim * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>distances</code> and <code>labels</code> must be non-null pointers and must satisfy the following:<a name="ul136865113167"></a><a name="ul136865113167"></a><ul id="ul136865113167"><li>When <code>merged = true</code>, their length must be <code>k * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li>When <code>merged = false</code>, their length must be <code>indexes.size() * k * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</li><li><code>filters</code> must be a <code>uint32_t</code> array of length <code>n * 6</code>; otherwise, out-of-bounds reads may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SearchWithFilter` (`AscendIndex`, Single Filter)<a name="en-us_TOPIC_0000001521894949"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2063619251347"><a name="p2063619251347"></a><a name="p2063619251347"></a><code>void SearchWithFilter(std::vector&lt;AscendIndex *&gt; indexes, idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, const void *filters, bool merged);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Searches multiple <code>indexes</code> and returns the IDs of the <code>k</code> most similar features based on the input feature vectors. It supports CID-based filtering. <code>filters</code> is a <code>uint32_t</code> array of length <code>n * 6</code>, and every six <code>uint32_t</code> values form one filter. The first four numbers of each filter (128 bits) represent the corresponding CID, and the last two numbers represent the left-closed timestamp range, that is, [<code>x</code>, <code>y</code>).</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p98111229152410"><a name="p98111229152410"></a><a name="p98111229152410"></a><code>std::vector&lt;AscendIndex *&gt; indexes</code>: Indexes to be used in the search.</p>
<p id="p311738102412"><a name="p311738102412"></a><a name="p311738102412"></a><code>idx_t n</code>: Number of queries to be used in the search.</p>
<p id="p41091246162416"><a name="p41091246162416"></a><a name="p41091246162416"></a><code>const float *x</code>: Query feature vectors to be used in the search.</p>
<p id="p17381952152416"><a name="p17381952152416"></a><a name="p17381952152416"></a><code>idx_t k</code>: Number of most similar results to return.</p>
<p id="p425111215144"><a name="p425111215144"></a><a name="p425111215144"></a><code>const void *filters</code>: Filter conditions.</p>
<p id="p11990043102211"><a name="p11990043102211"></a><a name="p11990043102211"></a><code>bool merged</code>: Whether to merge the results from search across multiple indexes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><code>float *distances</code>: Distance values between the query vectors and the <code>k</code> nearest vectors.</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><code>idx_t *labels</code>: IDs of the <code>k</code> nearest vectors to the queries.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul47662251713"></a><a name="ul47662251713"></a><ul id="ul47662251713"><li>The indexes participating in search must be created on the same card.</li><li>The supported <code>indexes</code> types are as follows.<a name="ul153841243520"></a><a name="ul153841243520"></a><ul id="ul153841243520"><li><code>indexes</code> points to <code>AscendIndexSQ</code>, and <code>QuantizerType</code> must be <code>QT_8bit</code>, with 0 &lt; <code>indexes.size()</code> ≤ 10000.</li><li><code>indexes</code> points to <code>AscendIndexIVFSP</code>, and the corresponding <code>QuantizerType</code> must be <code>QT_8bit</code> and <code>MetricType</code> must be <code>METRIC_L2</code>, with 0 &lt; <code>indexes.size()</code> ≤ 10000. The <code>AscendIndexIVFSP</code> indexes participating in search must share the same codebook in shared memory. You can create instances with the shared-codebook constructor provided by <code>AscendIndexIVFSP</code> or through the <code>loadAllData</code> interface.</li></ul>
</li></ul>
<a name="ul625373719595"></a><a name="ul625373719595"></a><ul id="ul625373719595"><li><code>n</code> must not exceed 1024.</li><li><code>k</code> must not exceed 1024.</li><li><code>x</code> must be a non-null pointer, and its length must be <code>dim * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>distances</code> and <code>labels</code> must be non-null pointers and must satisfy the following:<a name="ul136865113167"></a><a name="ul136865113167"></a><ul id="ul136865113167"><li>When <code>merged = true</code>, their length must be <code>k * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li>When <code>merged = false</code>, their length must be <code>indexes.size() * k * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</li><li><code>filters</code> must be a <code>uint32_t</code> array of length <code>n * 6</code>; otherwise, out-of-bounds reads may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SearchWithFilter` (`FaissIndex`, Multiple Filters)<a name="en-us_TOPIC_0000001635576093"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2063619251347"><a name="p2063619251347"></a><a name="p2063619251347"></a><code>void SearchWithFilter(std::vector&lt;Index *&gt; indexes, idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, void *filters[], bool merged);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p3393022191317"><a name="p3393022191317"></a><a name="p3393022191317"></a>Searches multiple <code>indexes</code> and returns the IDs of the <code>k</code> most similar features based on the input feature vectors.</p>
<p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>It supports CID-based filtering. <code>filters</code> is a pointer array of size <code>n</code>. Each pointer in the <code>filters</code> array points to a <code>uint32_t</code> array of size <code>indexes.size() * 6</code>. Every six <code>uint32_t</code> values form one filter. The first four numbers of each filter (128 bits) represent the corresponding CID, and the last two numbers represent the left-closed timestamp range, that is, [<code>x</code>, <code>y</code>).</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p98111229152410"><a name="p98111229152410"></a><a name="p98111229152410"></a><code>std::vector&lt;Index *&gt; indexes</code>: Indexes to be used in the search.</p>
<p id="p311738102412"><a name="p311738102412"></a><a name="p311738102412"></a><code>idx_t n</code>: Number of queries to be used in the search.</p>
<p id="p41091246162416"><a name="p41091246162416"></a><a name="p41091246162416"></a><code>const float *x</code>: Query feature vectors to be used in the search.</p>
<p id="p17381952152416"><a name="p17381952152416"></a><a name="p17381952152416"></a><code>idx_t k</code>: Number of most similar results to return.</p>
<p id="p425111215144"><a name="p425111215144"></a><a name="p425111215144"></a><code>void *filters[]</code>: Filter conditions.</p>
<p id="p11990043102211"><a name="p11990043102211"></a><a name="p11990043102211"></a><code>bool merged</code>: Whether to merge the results from search across multiple indexes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p15894943184510"><a name="p15894943184510"></a><a name="p15894943184510"></a><code>float *distances</code>: Distance values between the query vectors and the <code>k</code> nearest vectors.</p>
<p id="p6295973819"><a name="p6295973819"></a><a name="p6295973819"></a><code>idx_t *labels</code>: IDs of the <code>k</code> nearest vectors to the queries.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul151271598168"></a><a name="ul151271598168"></a><ul id="ul151271598168"><li>The indexes participating in search must be created on the same card.</li></ul>
<a name="ul730213891116"></a><a name="ul730213891116"></a><ul id="ul730213891116"><li>The supported <code>indexes</code> types are as follows.<a name="ul11318141395713"></a><a name="ul11318141395713"></a><ul id="ul11318141395713"><li><code>indexes</code> points to <code>AscendIndexSQ</code>, and <code>QuantizerType</code> must be <code>QT_8bit</code>, with 0 &lt; <code>indexes.size()</code> ≤ 10000.</li><li><code>indexes</code> points to <code>AscendIndexIVFSP</code>, and the corresponding <code>QuantizerType</code> must be <code>QT_8bit</code> and <code>MetricType</code> must be <code>METRIC_L2</code>, with 0 &lt; <code>indexes.size()</code> ≤ 10000. The <code>AscendIndexIVFSP</code> indexes participating in search must share the same codebook in shared memory. You can create instances with the shared-codebook constructor provided by <code>AscendIndexIVFSP</code> or through the <code>loadAllData</code> interface.</li></ul>
</li></ul>
<a name="ul625373719595"></a><a name="ul625373719595"></a><ul id="ul625373719595"><li><code>n</code> must not exceed 1024.</li><li><code>k</code> must not exceed 1024.</li><li><code>x</code> must be a non-null pointer, and its length must be <code>dim * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>distances</code> and <code>labels</code> must be non-null pointers and must satisfy the following:<a name="ul136865113167"></a><a name="ul136865113167"></a><ul id="ul136865113167"><li>When <code>merged = true</code>, their length must be <code>k * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li>When <code>merged = false</code>, their length must be <code>indexes.size() * k * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</li><li><code>filters</code> must be a pointer array of length <code>n</code>, and each pointer in the array must point to a <code>uint32_t</code> array of length <code>indexes.size() * 6</code>; otherwise, out-of-bounds reads may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SearchWithFilter` (`AscendIndex`, Multiple Filters)<a name="en-us_TOPIC_0000001635815493"></a>

<a name="table20177631161415"></a>
<table><tbody><tr id="row141771631111420"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p1017733121417"><a name="p1017733121417"></a><a name="p1017733121417"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1817723161412"><a name="p1817723161412"></a><a name="p1817723161412"></a><code>void SearchWithFilter(std::vector&lt;AscendIndex *&gt; indexes, idx_t n, const float *x, idx_t k, float *distances, idx_t *labels, void *filters[], bool merged);</code></p>
</td>
</tr>
<tr id="row141772313143"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p517753151420"><a name="p517753151420"></a><a name="p517753151420"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p734718314137"><a name="p734718314137"></a><a name="p734718314137"></a>Searches multiple <code>indexes</code> and returns the IDs of the <code>k</code> most similar features based on the input feature vectors.</p>
<p id="p91771731111417"><a name="p91771731111417"></a><a name="p91771731111417"></a>It supports CID-based filtering. <code>filters</code> is a pointer array of size <code>n</code>. Each pointer in the <code>filters</code> array points to a <code>uint32_t</code> array of size <code>indexes.size() * 6</code>. Every six <code>uint32_t</code> values form one filter. The first four numbers of each filter (128 bits) represent the corresponding CID, and the last two numbers represent the left-closed timestamp range, that is, [<code>x</code>, <code>y</code>).</p>
</td>
</tr>
<tr id="row121771315147"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p141779311146"><a name="p141779311146"></a><a name="p141779311146"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p19177173111418"><a name="p19177173111418"></a><a name="p19177173111418"></a><code>std::vector&lt;AscendIndex *&gt; indexes</code>: Indexes to be used in the search.</p>
<p id="p141771319144"><a name="p141771319144"></a><a name="p141771319144"></a><code>idx_t n</code>: Number of queries to be used in the search.</p>
<p id="p1217711314143"><a name="p1217711314143"></a><a name="p1217711314143"></a><code>const float *x</code>: Query feature vectors to be used in the search.</p>
<p id="p0177133112144"><a name="p0177133112144"></a><a name="p0177133112144"></a><code>idx_t k</code>: Number of most similar results to return.</p>
<p id="p1417863120145"><a name="p1417863120145"></a><a name="p1417863120145"></a><code>void *filters[]</code>: Filter conditions.</p>
<p id="p1217823110149"><a name="p1217823110149"></a><a name="p1217823110149"></a><code>bool merged</code>: Whether to merge the results from search across multiple indexes.</p>
</td>
</tr>
<tr id="row19178531191410"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p9178113114145"><a name="p9178113114145"></a><a name="p9178113114145"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p717803161415"><a name="p717803161415"></a><a name="p717803161415"></a><code>float *distances</code>: Distance values between the query vectors and the <code>k</code> nearest vectors.</p>
<p id="p10178133101412"><a name="p10178133101412"></a><a name="p10178133101412"></a><code>idx_t *labels</code>: IDs of the <code>k</code> nearest vectors to the queries.</p>
</td>
</tr>
<tr id="row1217833112143"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p131788313148"><a name="p131788313148"></a><a name="p131788313148"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1417819317143"><a name="p1417819317143"></a><a name="p1417819317143"></a>None</p>
</td>
</tr>
<tr id="row3178133111141"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p217843111416"><a name="p217843111416"></a><a name="p217843111416"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul151271598168"></a><a name="ul151271598168"></a><ul id="ul151271598168"><li>The indexes participating in search must be created on the same card.</li></ul>
<a name="ul671038151117"></a><a name="ul671038151117"></a><ul id="ul671038151117"><li>The supported <code>indexes</code> types are as follows.<a name="ul11318141395713"></a><a name="ul11318141395713"></a><ul id="ul11318141395713"><li><code>indexes</code> points to <code>AscendIndexSQ</code>, and <code>QuantizerType</code> must be <code>QT_8bit</code>, with 0 &lt; <code>indexes.size()</code> ≤ 10000.</li><li><code>indexes</code> points to <code>AscendIndexIVFSP</code>, and the corresponding <code>QuantizerType</code> must be <code>QT_8bit</code> and <code>MetricType</code> must be <code>METRIC_L2</code>, with 0 &lt; <code>indexes.size()</code> ≤ 10000. The <code>AscendIndexIVFSP</code> indexes participating in search must share the same codebook in shared memory. You can create instances with the shared-codebook constructor provided by <code>AscendIndexIVFSP</code> or through the <code>loadAllData</code> interface.</li></ul>
</li></ul>
<a name="ul51781331161411"></a><a name="ul51781331161411"></a><ul id="ul51781331161411"><li><code>n</code> must not exceed 1024.</li><li><code>k</code> must not exceed 1024.</li><li><code>x</code> must be a non-null pointer, and its length must be <code>dim * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li><code>distances</code> and <code>labels</code> must be non-null pointers and must satisfy the following:<a name="ul136865113167"></a><a name="ul136865113167"></a><ul id="ul136865113167"><li>When <code>merged = true</code>, their length must be <code>k * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li><li>When <code>merged = false</code>, their length must be <code>indexes.size() * k * n</code>; otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</li><li><code>filters</code> must be a pointer array of length <code>n</code>, and each pointer in the array must point to a <code>uint32_t</code> array of length <code>indexes.size() * 6</code>; otherwise, out-of-bounds reads may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>
