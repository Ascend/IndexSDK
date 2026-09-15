# `IndexIL`<a name="en-us_TOPIC_0000001506414825"></a>

## Function Description<a name="en-us_TOPIC_0000001456535188"></a>

`IndexIL` is a feature management abstract class based on a contiguous memory allocation mechanism. It serves retrieval algorithms that use indices as labels. To use it, you must inherit from it and implement all interfaces.

The vectors stored in the base library and the query vectors of each API must be normalized FP16 floating-point values. (`IL` stands for "Indices as Labels".)

It does not support multithreaded concurrent calls. Therefore, in multithreaded scenarios, the user must lock before use. Otherwise, the retrieval APIs may raise exceptions. It also does not support sharing a Device across different threads.

## `AddFeatures`<a name="en-us_TOPIC_0000001506414693"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>virtual APP_ERROR AddFeatures(int n, const float16_t *features, const idx_t *indices) = 0;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Inserts <code>n</code> feature vectors with specified indices into the feature library. If a feature vector already exists at an index, this insertion is equivalent to an update.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b6166642122410"><a name="b6166642122410"></a><a name="b6166642122410"></a><code>int n</code></strong>: The number of feature vectors to insert.</p>
<p id="p19117872412"><a name="p19117872412"></a><a name="p19117872412"></a><strong id="b61461844172410"><a name="b61461844172410"></a><a name="b61461844172410"></a><code>const float16_t *features</code></strong>: Feature vectors, with a length of <code>n * vector dimension dim</code>.</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b416612460248"><a name="b416612460248"></a><a name="b416612460248"></a><code>const idx_t *indices</code></strong>: The index values corresponding to the feature vectors, with a length of <code>n</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The return status of the call. For details, see the reference for API return values.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul7288226205219"></a><a name="ul7288226205219"></a><ul id="ul7288226205219"><li>The input parameters are constrained by the implementation class.</li><li><code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `IndexIL`<a name="en-us_TOPIC_0000001456695020"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="21.240000000000002%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="78.75999999999999%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>IndexIL();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="21.240000000000002%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="78.75999999999999%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>The constructor of <code>IndexIL</code>. It creates a feature management object.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="21.240000000000002%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="78.75999999999999%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="21.240000000000002%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="78.75999999999999%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="21.240000000000002%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="78.75999999999999%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="21.240000000000002%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="78.75999999999999%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `~IndexIL`<a name="en-us_TOPIC_0000001506334781"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>virtual ~IndexIL();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>The destructor of <code>IndexIL</code>. It destroys the feature management object.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>None</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `Finalize`<a name="en-us_TOPIC_0000001456375356"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1997210581407"><a name="p1997210581407"></a><a name="p1997210581407"></a><code>virtual APP_ERROR Finalize() = 0;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Releases the feature library management resources.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b9837134732319"><a name="b9837134732319"></a><a name="b9837134732319"></a><code>APP_ERROR</code></strong>: The return status of the call. For details, see the reference for API return values.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `GetFeatures`<a name="en-us_TOPIC_0000001506495833"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>virtual APP_ERROR GetFeatures(int n, float16_t *features, const idx_t *indices) = 0;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Queries the feature vectors for <code>n</code> specified indices.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p19117872412"><a name="p19117872412"></a><a name="p19117872412"></a><strong id="b7444782517"><a name="b7444782517"></a><a name="b7444782517"></a><code>int n</code></strong>: The number of feature vectors to obtain.</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b144851412192517"><a name="b144851412192517"></a><a name="b144851412192517"></a><code>const idx_t *indices</code></strong>: The index values to query, with a length of <code>n</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p81034246387"><a name="p81034246387"></a><a name="p81034246387"></a><strong id="b16464142810254"><a name="b16464142810254"></a><a name="b16464142810254"></a><code>float16_t *features</code></strong>: The feature vectors corresponding to the queried indices, with a length of <code>n * vector dimension dim</code>. The user must allocate memory before the call and ensure that the memory size is correct.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b1557493018253"><a name="b1557493018253"></a><a name="b1557493018253"></a><code>APP_ERROR</code></strong>: The return status of the call. For details, see the reference for API return values.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul7288226205219"></a><a name="ul7288226205219"></a><ul id="ul7288226205219"><li>The input parameters are constrained by the implementation class.</li><li><code>features</code> and <code>indices</code> must be non-null pointers, and their lengths must meet the constraints. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `GetNTotal`<a name="en-us_TOPIC_0000001456535092"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>virtual int GetNTotal() const = 0;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p189931428115213"><a name="p189931428115213"></a><a name="p189931428115213"></a>Queries the maximum occupied space of the current feature library vectors.</p>
<p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Feature vectors are inserted starting from index <strong id="b320263895212"><a name="b320263895212"></a><a name="b320263895212"></a><code>0</code></strong>. If the inserted feature vector indices are continuous, <code>ntotal</code> equals the number of feature vectors. Otherwise, <code>ntotal</code> equals the maximum inserted index value plus <code>1</code>. For performance reasons, the operator batches memory operations and, by default, treats the space at and before the maximum index position as valid base library vectors and includes it in the calculation. The user must use this API to obtain the total number of base library entries recorded inside the <code>Index</code>, and then allocate the corresponding memory space to pass parameters to the corresponding functional APIs. For details, see the specific API.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p972735784416"><a name="p972735784416"></a><a name="p972735784416"></a><strong id="b4727557174419"><a name="b4727557174419"></a><a name="b4727557174419"></a><code>int ntotal</code></strong>: See the description.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `Init`<a name="en-us_TOPIC_0000001506334657"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1939248184018"><a name="p1939248184018"></a><a name="p1939248184018"></a><code>virtual APP_ERROR Init(int dim, int capacity, AscendMetricType metricType, int64_t resourceSize) = 0;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p18122312578"><a name="p18122312578"></a><a name="p18122312578"></a>Initializes feature library parameters and allocates base library memory resources.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b156115254222"><a name="b156115254222"></a><a name="b156115254222"></a><code>int dim</code></strong>: Feature vector dimension.</p>
<p id="p1889154465814"><a name="p1889154465814"></a><a name="p1889154465814"></a><strong id="b0291530172212"><a name="b0291530172212"></a><a name="b0291530172212"></a><code>AscendMetricType metricType</code></strong>: Feature distance type, including inner product, Euclidean distance, and cosine similarity.</p>
<p id="p45951117599"><a name="p45951117599"></a><a name="p45951117599"></a><strong id="b8478173318223"><a name="b8478173318223"></a><a name="b8478173318223"></a><code>int capacity</code></strong>: Maximum base library capacity. The allocated memory size is <code>capacity * dim * sizeof(float)</code> bytes.</p>
<p id="p12851134435520"><a name="p12851134435520"></a><a name="p12851134435520"></a><strong id="b1968193195310"><a name="b1968193195310"></a><a name="b1968193195310"></a><code>int resourceSize</code></strong>: Preallocates Device-side cache resources. When a retrieval API is called, it can use these resources directly instead of calling <code>aclrtmalloc</code> to allocate memory, which improves performance. The default value is <code>-1</code>, which means the cache resource is allocated with the default size of <code>128 MB</code>. You can configure the actual size more precisely based on the retrieval workload and Device-side resource usage.</p>
<p id="p1703214386"><a name="p1703214386"></a><a name="p1703214386"></a>For example, if the query batch size is <code>64</code>, the base library contains 1,000,000 vectors, and one FP32 value occupies 4 bytes, set <code>resourceSize</code> to <code>64 * 1000000 * 4 = 256,000,000</code> bytes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b18120185612228"><a name="b18120185612228"></a><a name="b18120185612228"></a><code>APP_ERROR</code></strong>: The return status of the call. For details, see the reference for API return values.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p392515311255"><a name="p392515311255"></a><a name="p392515311255"></a>The input parameters are constrained by the implementation class.</p>
</td>
</tr>
</tbody>
</table>

## `RemoveFeatures`<a name="en-us_TOPIC_0000001456534932"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>virtual APP_ERROR RemoveFeatures(int n, const idx_t *indices) = 0;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Deletes <code>n</code> feature vectors with specified indices from the vector library.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p19117872412"><a name="p19117872412"></a><a name="p19117872412"></a><strong id="b7444782517"><a name="b7444782517"></a><a name="b7444782517"></a><code>int n</code></strong>: The number of feature vectors to delete.</p>
<p id="p1672132542420"><a name="p1672132542420"></a><a name="p1672132542420"></a><strong id="b144851412192517"><a name="b144851412192517"></a><a name="b144851412192517"></a><code>const idx_t *indices</code></strong>: The indices of the feature vectors to delete.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a><strong id="b2046121817254"><a name="b2046121817254"></a><a name="b2046121817254"></a><code>APP_ERROR</code></strong>: The return status of the call. For details, see the reference for API return values.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul7288226205219"></a><a name="ul7288226205219"></a><ul id="ul7288226205219"><li>The input parameters are constrained by the implementation class.</li><li><code>indices</code> must be a non-null pointer, and its length must be <code>n</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SetNTotal`<a name="en-us_TOPIC_0000001456375256"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>virtual APP_ERROR SetNTotal(int n) = 0;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p57481224513"><a name="p57481224513"></a><a name="p57481224513"></a>Provides an interface for adjusting the <code>ntotal</code> count externally.</p>
<p id="p16965727122812"><a name="p16965727122812"></a><a name="p16965727122812"></a>After base library vectors are added, the <code>Index</code> internally updates the <code>ntotal</code> value according to the largest inserted index, but it does not record which regions in the range <code>[0, ntotal]</code> are invalid. Therefore, the <code>RemoveFeatures</code> operation does not change the <code>ntotal</code> value. If you explicitly record the maximum base library index after insert and delete operations on the service side, you can set <code>ntotal</code> manually. This reduces the operator workload within a controllable range and improves interface performance.</p>
<p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>For example, if 100 vectors are inserted and the base library indices range from <code>0</code> to <code>99</code>, <code>ntotal = 100</code>. If you delete the base library entries with indices from <code>80</code> to <code>90</code>, the <code>ntotal</code> value inside the <code>Index</code> remains unchanged and can only be set to a value in <code>[ntotal, capacity]</code>. If you then delete the base library entries with indices from <code>90</code> to <code>99</code>, you can manually set <code>ntotal</code> to a value in <code>[80, capacity]</code>. When you set it to <code>80</code>, the amount of base library data involved in comparison decreases by 20 vectors.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b8131171512139"><a name="b8131171512139"></a><a name="b8131171512139"></a><code>int n</code></strong>: The maximum base library index managed by the service side, plus <code>1</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p432242682918"><a name="p432242682918"></a><a name="p432242682918"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The return status of the call. For details, see the reference for API return values.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>The input parameters are constrained by the implementation class.</p>
</td>
</tr>
</tbody>
</table>
