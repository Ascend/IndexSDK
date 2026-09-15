# DiskPQ<a name="en-us_TOPIC_0000002382802364"></a>

## Function Description<a name="en-us_TOPIC_0000002382647580"></a>

Index SDK provides training and retrieval functions for PQ (Product Quantization). The PQ interface does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, you need to lock before use. Otherwise, the function may behave abnormally.

## `DiskPQParams`<a name="en-us_TOPIC_0000002382807444"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p784562919563"><a name="p784562919563"></a><a name="p784562919563"></a><code>DiskPQParams {</code></p>
<p id="p584518296562"><a name="p584518296562"></a><a name="p584518296562"></a><code>int pqChunks = 512;</code></p>
<p id="p17845192912562"><a name="p17845192912562"></a><a name="p17845192912562"></a><code>int funcType = 1;</code></p>
<p id="p1884502910567"><a name="p1884502910567"></a><a name="p1884502910567"></a><code>int dim = 1;</code></p>
<p id="p8845122945614"><a name="p8845122945614"></a><a name="p8845122945614"></a><code>char *pqTable = nullptr;</code></p>
<p id="p13845202914564"><a name="p13845202914564"></a><a name="p13845202914564"></a><code>uint32_t *offsets = nullptr;</code></p>
<p id="p28451029185617"><a name="p28451029185617"></a><a name="p28451029185617"></a><code>char *tablesTransposed = nullptr;</code></p>
<p id="p12845102915560"><a name="p12845102915560"></a><a name="p12845102915560"></a><code>char *centroids = nullptr;</code></p>
<p id="p584532905613"><a name="p584532905613"></a><a name="p584532905613"></a><code>}</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p139815474417"><a name="p139815474417"></a><a name="p139815474417"></a>PQ quantization structure.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1639825419446"><a name="p1639825419446"></a><a name="p1639825419446"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p203971254164419"><a name="p203971254164419"></a><a name="p203971254164419"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Parameter Values</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p11645124317919"><a name="p11645124317919"></a><a name="p11645124317919"></a><code>int pqChunks</code>: Splits the original vector dimension <code>dim</code> into <code>pqChunks</code> chunks.</p>
<p id="p1264516437919"><a name="p1264516437919"></a><a name="p1264516437919"></a><code>int funcType</code>: Computation standard used for PQ table lookup distance calculation.</p>
<p id="p564519434918"><a name="p564519434918"></a><a name="p564519434918"></a><code>int dim</code>: Original data dimension.</p>
<p id="p164518432910"><a name="p164518432910"></a><a name="p164518432910"></a><code>char *pqTable</code>: Pointer to the codebook data. The default value is <code>nullptr</code>.</p>
<p id="p164510431912"><a name="p164510431912"></a><a name="p164510431912"></a><code>uint32_t *offsets</code>: Pointer to the starting and ending dimensions of each chunk in the original dimension. The default value is <code>nullptr</code>.</p>
<p id="p16645243598"><a name="p16645243598"></a><a name="p16645243598"></a><code>char *tablesTransposed</code>: Pointer to the transposed form of the codebook data. The default value is <code>nullptr</code>.</p>
<p id="p864518435917"><a name="p864518435917"></a><a name="p864518435917"></a><code>char *centroids</code>: Pointer to the mean value of each dimension, used to center the data. The default value is <code>nullptr</code>.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Parameter Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1368219905119"></a><a name="ul1368219905119"></a><ul id="ul1368219905119"><li>1 &lt;= <code>pqChunks</code> &lt;= <code>dim</code>. Smaller <code>pqChunks</code> use less memory, but they also reduce accuracy. In general, you are advised to set <code>pqChunks</code> to <code>dim / 8</code> or <code>dim / 16</code>, rounded up in both cases. The default value is 512.</li><li>The valid range of <code>funcType</code> is 1 to 3. 1 indicates L2 distance, 2 indicates IP distance, and 3 indicates cosine distance. The default value is 1.</li><li>1 &lt;= <code>dim</code> &lt;= 2000. The default value is 1.</li><li><code>pqTable</code> currently supports only the <code>float</code> data type, that is, the <code>Vector</code> data type in OpenGauss.</li><li><code>tablesTransposed</code> currently supports only the <code>float</code> data type, that is, the <code>Vector</code> data type in OpenGauss.</li></ul>
</td>
</tr>
</tbody>
</table>

## `VectorArrayData`<a name="en-us_TOPIC_0000002416326913"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p455275425615"><a name="p455275425615"></a><a name="p455275425615"></a><code>VectorArrayData {</code></p>
<p id="p055225485618"><a name="p055225485618"></a><a name="p055225485618"></a><code>int length;</code></p>
<p id="p20552454105617"><a name="p20552454105617"></a><a name="p20552454105617"></a><code>int maxlen;</code></p>
<p id="p1355211547565"><a name="p1355211547565"></a><a name="p1355211547565"></a><code>int dim;</code></p>
<p id="p15552155445611"><a name="p15552155445611"></a><a name="p15552155445611"></a><code>size_t itemsize;</code></p>
<p id="p655255413561"><a name="p655255413561"></a><a name="p655255413561"></a><code>char *items;</code></p>
<p id="p75521654195610"><a name="p75521654195610"></a><a name="p75521654195610"></a><code>}</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p139815474417"><a name="p139815474417"></a><a name="p139815474417"></a>Data encapsulation structure.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1639825419446"><a name="p1639825419446"></a><a name="p1639825419446"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p203971254164419"><a name="p203971254164419"></a><a name="p203971254164419"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Parameter Values</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p11645124317919"><a name="p11645124317919"></a><a name="p11645124317919"></a><code>int length</code>: Number of vectors stored in the structure.</p>
<p id="p1264516437919"><a name="p1264516437919"></a><a name="p1264516437919"></a><code>int maxlen</code>: Maximum number of vectors stored in the structure.</p>
<p id="p564519434918"><a name="p564519434918"></a><a name="p564519434918"></a><code>int dim</code>: Vector dimension stored in the structure.</p>
<p id="p164518432910"><a name="p164518432910"></a><a name="p164518432910"></a><code>size_t itemsize</code>: Reserved field. Users can choose not to set it.</p>
<p id="p164510431912"><a name="p164510431912"></a><a name="p164510431912"></a><code>char *items</code>: Pointer to the data stored in <code>VectorArrayData</code>. The default value is <code>nullptr</code>.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Parameter Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1368219905119"></a><a name="ul1368219905119"></a><ul id="ul1368219905119"><li>1 &lt;= <code>length</code> &lt;= 100000000.</li><li><code>maxlen</code> is a reserved field on the OpenGauss side. Non-OpenGauss users can set it to the same value as <code>length</code>.</li><li>1 &lt;= <code>dim</code> &lt;= 2000.</li><li>For different APIs, ensure that <code>items</code> points to data of the required size.</li></ul>
</td>
</tr>
</tbody>
</table>

## `ComputePQTable`<a name="en-us_TOPIC_0000002416446741"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a><code>int ComputePQTable(VectorArrayData *sample, DiskPQParams *params);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>Uses the sampled base-library data stored in <code>sample</code> to compute the PQ codebook and stores the codebook-related data in the corresponding parameters in <code>params</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12607112054318"><a name="p12607112054318"></a><a name="p12607112054318"></a><code>VectorArrayData *sample</code>: Pointer to the <code>VectorArrayData</code> instance that contains the sampled base-library data. Must not be a null pointer.</p>
<p id="p8607152012439"><a name="p8607152012439"></a><a name="p8607152012439"></a><code>DiskPQParams *params</code>: Pointer to the <code>DiskPQParams</code> instance that contains only PQ parameters and no trained PQ data. Must not be a null pointer.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p116514331439"><a name="p116514331439"></a><a name="p116514331439"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a><code>int</code>: <code>0</code> indicates that the process is normal. <code>-1</code> indicates that the process failed, and the error logs are printed to <code>cerr</code>.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1693011362444"></a><a name="ul1693011362444"></a><ul id="ul1693011362444"><li>The <code>sample</code> data must meet the following requirements:<p id="p14941713111016"><a name="p14941713111016"></a><a name="p14941713111016"></a>The data pointed to by <code>items</code> must be <code>(8 + dim) * length * sizeof(float)</code> bytes, which means each vector has 8 bytes of metadata in front of it. When non-OpenGauss users use this API, they need to add 8 bytes of arbitrary data to each vector entry.</p>
</li><li>The <code>params</code> members must meet the following requirements:<a name="ul156691381133"></a><a name="ul156691381133"></a><ul id="ul156691381133"><li>In addition to the range limits described above, <code>dim</code> must match the corresponding <code>dim</code> field in <code>sample</code>.</li><li><code>pqTable</code> must be <code>nullptr</code>. The dynamic library allocates memory with <code>new []</code>, and you must release the allocated memory outside the library with <code>delete []</code>. The allocated memory size is <code>dim * 256 * sizeof(float)</code> bytes, where 256 is the number of clusters in each chunk.</li><li><code>offsets</code> must be <code>nullptr</code>. The dynamic library allocates memory with <code>new []</code>, and you must release the allocated memory outside the library with <code>delete []</code>. The allocated memory size is <code>(pqChunks + 1) * sizeof(uint32_t)</code> bytes.</li><li><code>tablesTransposed</code> must be <code>nullptr</code>. The dynamic library allocates memory with <code>new []</code>, and you must release the allocated memory outside the library with <code>delete []</code>. The allocated memory size is <code>dim * 256 * sizeof(float)</code> bytes.</li><li><code>centroids</code> must be <code>nullptr</code>. The dynamic library allocates memory with <code>new []</code>, and you must release the allocated memory outside the library with <code>delete []</code>. The allocated memory size is <code>dim * sizeof(float)</code> bytes.</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

## `ComputeVectorPQCode`<a name="en-us_TOPIC_0000002382647584"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p953585410711"><a name="p953585410711"></a><a name="p953585410711"></a><code>int ComputeVectorPQCode(VectorArrayData *baseData, const DiskPQParams *params, uint8_t *pqCode);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>Uses the <code>params</code> filled with PQ data to quantize the base-library data in <code>baseData</code> and writes the quantized data into the buffer pointed to by <code>pqCode</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12607112054318"><a name="p12607112054318"></a><a name="p12607112054318"></a><code>VectorArrayData *baseData</code>: Pointer to the <code>VectorArrayData</code> instance that contains the base-library data. Must not be a null pointer. You can determine the size of the base-library data in <code>baseData</code> externally based on your memory limits.</p>
<p id="p8607152012439"><a name="p8607152012439"></a><a name="p8607152012439"></a><code>const DiskPQParams *params</code>: Pointer to the <code>DiskPQParams</code> instance that contains PQ parameters and trained PQ data. Must not be a null pointer.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p116514331439"><a name="p116514331439"></a><a name="p116514331439"></a><code>uint8_t *pqCode</code>: Pointer that receives the compressed base-library vectors. Must not be a null pointer.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a><code>int</code>: <code>0</code> indicates that the process is normal. <code>-1</code> indicates that the process failed, and the error logs are printed to <code>cerr</code>.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1693011362444"></a><a name="ul1693011362444"></a><ul id="ul1693011362444"><li>The <code>baseData</code> data must meet the following requirements:<p id="p71308396175"><a name="p71308396175"></a><a name="p71308396175"></a>The data pointed to by <code>items</code> must be <code>length * dim * sizeof(float)</code> bytes. Note that, unlike the <code>ComputePQTable</code> interface, you do not need to add placeholder metadata before each data entry.</p>
</li><li>The <code>params</code> members must meet the following requirements:<a name="ul156691381133"></a><a name="ul156691381133"></a><ul id="ul156691381133"><li>In addition to the range limits described above, <code>dim</code> must match the corresponding <code>dim</code> field in <code>baseData</code>.</li><li><code>pqTable</code> must point to codebook data whose size is <code>dim * 256 * sizeof(float)</code> bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur.</li><li><code>offsets</code> must point to <code>offsets</code> data whose size is <code>(pqChunks + 1) * sizeof(uint32_t)</code> bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur.</li><li>There is no requirement for <code>tablesTransposed</code>.</li><li><code>centroids</code> must point to <code>centroids</code> data whose size is <code>dim * sizeof(float)</code> bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur.</li></ul>
</li><li>Ensure that the space pointed to by <code>pqCode</code> is at least <code>length * pqChunks</code> bytes. Here, <code>length</code> is the <code>VectorArrayData</code> parameter and <code>pqChunks</code> is the <code>DiskPQParams</code> parameter.</li></ul>
</td>
</tr>
</tbody>
</table>

## `GetPQDistanceTable`<a name="en-us_TOPIC_0000002382807448"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p953585410711"><a name="p953585410711"></a><a name="p953585410711"></a><code>int GetPQDistanceTable(char *vec, const DiskPQParams *params, float *pqDistanceTable);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>Uses the <code>params</code> filled with PQ data to perform ADC PQ distance calculation on the query data pointed to by <code>vec</code> and writes the PQ distance table into the buffer pointed to by <code>pqDistanceTable</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p12607112054318"><a name="p12607112054318"></a><a name="p12607112054318"></a><code>char *vec</code>: Pointer to the query data to calculate.</p>
<p id="p86231040131614"><a name="p86231040131614"></a><a name="p86231040131614"></a><code>const DiskPQParams *params</code>: Pointer to the <code>DiskPQParams</code> instance that contains PQ parameters and trained PQ data. Must not be a null pointer.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p116514331439"><a name="p116514331439"></a><a name="p116514331439"></a><code>float *pqDistanceTable</code>: Pointer that receives the distances between the query and each centroid in each chunk.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a><code>int</code>: <code>0</code> indicates that the process is normal. <code>-1</code> indicates that the process failed, and the error logs are printed to <code>cerr</code>.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1693011362444"></a><a name="ul1693011362444"></a><ul id="ul1693011362444"><li>Ensure that the space pointed to by <code>vec</code> is at least <code>dim * sizeof(float)</code> bytes. Currently, only the <code>float</code> data type is supported, that is, the <code>Vector</code> data type in OpenGauss.</li><li>The <code>params</code> members must meet the following requirements:<a name="ul156691381133"></a><a name="ul156691381133"></a><ul id="ul156691381133"><li>There is no requirement for the value pointed to by <code>pqTable</code>.</li><li><code>offsets</code> must point to <code>offsets</code> data whose size is <code>(pqChunks + 1) * sizeof(uint32_t)</code> bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur.</li><li><code>tablesTransposed</code> must point to codebook data whose size is <code>dim * 256 * sizeof(float)</code> bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur.</li><li><code>centroids</code> must point to <code>centroids</code> data whose size is <code>dim * sizeof(float)</code> bytes. Ensure that the memory size pointed to is valid, or a segmentation fault may occur.</li></ul>
</li><li>Ensure that the space pointed to by <code>pqDistanceTable</code> is at least <code>pqChunks * 256 * sizeof(float)</code> bytes.</li></ul>
</td>
</tr>
</tbody>
</table>

## `GetPQDistance`<a name="en-us_TOPIC_0000002416326917"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p953585410711"><a name="p953585410711"></a><a name="p953585410711"></a><code>int GetPQDistance(const uint8_t *basecode, const DiskPQParams *params, const float *pqDistanceTable, float &amp;pqDistance);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p169090129309"><a name="p169090129309"></a><a name="p169090129309"></a>Uses the compressed code data corresponding to the base-library vector pointed to by <code>basecode</code> and the <code>pqDistanceTable</code> obtained from the <code>GetPQDistanceTable</code> API to calculate the PQ distance between the query and that base-library vector.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1533165363012"><a name="p1533165363012"></a><a name="p1533165363012"></a><code>const uint8_t *basecode</code>: Pointer to the compressed code data corresponding to a base-library vector.</p>
<p id="p12607112054318"><a name="p12607112054318"></a><a name="p12607112054318"></a><code>const DiskPQParams *params</code>: Pointer to the <code>DiskPQParams</code> instance with the <code>pqChunks</code> value filled in. Must not be a null pointer.</p>
<p id="p18804047171617"><a name="p18804047171617"></a><a name="p18804047171617"></a><code>const float *pqDistanceTable</code>: Pointer to the ADC PQ distance table corresponding to the query.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p116514331439"><a name="p116514331439"></a><a name="p116514331439"></a><code>float &amp;pqDistance</code>: Reference to the final output PQ distance value.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a><code>int</code>: <code>0</code> indicates that the process is normal. <code>-1</code> indicates that the process failed, and the error logs are printed to <code>cerr</code>.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1693011362444"></a><a name="ul1693011362444"></a><ul id="ul1693011362444"><li>Ensure that the data pointed to by <code>basecode</code> is at least <code>pqChunks</code> bytes.</li><li>In <code>params</code>, you only need to fill in the <code>pqChunks</code> value, and it must match the <code>pqChunks</code> value mentioned for <code>basecode</code>.</li><li>Ensure that the data pointed to by <code>pqDistanceTable</code> is at least <code>pqChunks * 256 * sizeof(float)</code> bytes.</li><li>The interface does not zero <code>pqDistance</code> before use. The final result of <code>pqDistance</code> is the original <code>pqDistance</code> value plus the PQ distance between the query and <code>basecode</code>. Therefore, an input value of <code>0</code> is recommended.</li></ul>
</td>
</tr>
</tbody>
</table>
