# `AscendCloner`<a name="en-us_TOPIC_0000001506334577"></a>

## Function Description<a name="en-us_TOPIC_0000001456375412"></a>

Index SDK provides an operation that copies retrieval `Index` resources on the NPU to Faiss on the CPU side. The copy process happens in memory. Data loaded in the original NPU `Index` is copied into CPU-side memory, which makes it convenient for users to run retrieval with the same base library on the CPU.

> [!NOTE]
> Some versions of Faiss provide a method for persisting an in-memory `Index` to disk, that is, saving in-memory data to a local drive. When you use Index SDK and Faiss to process sensitive data, pay special attention to the corresponding access control and encryption protection.

## `index_ascend_to_cpu`<a name="en-us_TOPIC_0000001506334821"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p01708458179"><a name="p01708458179"></a><a name="p01708458179"></a><code>faiss::Index *index_ascend_to_cpu(const faiss::Index *ascend_index);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Copies retrieval `Index` resources on Ascend and creates a retrieval `Index` on the CPU.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p21724117137"><a name="p21724117137"></a><a name="p21724117137"></a><code>const faiss::Index *ascend_index</code>: `Index` resource on Ascend.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1246923411510"><a name="p1246923411510"></a><a name="p1246923411510"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>A retrieval `Index` on the CPU.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p3947640162619"><a name="p3947640162619"></a><a name="p3947640162619"></a>After you finish using the returned `Index` pointer, remember to delete this pointer to release the corresponding memory.</p>
</td>
</tr>
</tbody>
</table>

## `index_cpu_to_ascend`<a name="en-us_TOPIC_0000001456695032"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1915092217197"><a name="p1915092217197"></a><a name="p1915092217197"></a><code>faiss::Index *index_cpu_to_ascend(std::initializer_list&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Copies retrieval `Index` resources on the CPU and creates a retrieval `Index` on Ascend.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p21724117137"><a name="p21724117137"></a><a name="p21724117137"></a><code>std::initializer_list&lt;int&gt; devices</code>: Device IDs to configure on the NPU.</p>
<p id="p142883052016"><a name="p142883052016"></a><a name="p142883052016"></a><code>const faiss::Index *index</code>: Retrieval `Index` resources on the CPU.</p>
<p id="p15735115410199"><a name="p15735115410199"></a><a name="p15735115410199"></a><code>const AscendClonerOptions *options = nullptr</code>: `AscendClonerOptions` resource to configure.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1246923411510"><a name="p1246923411510"></a><a name="p1246923411510"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>A retrieval `Index` on Ascend.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul37673419536"></a><a name="ul37673419536"></a><ul id="ul37673419536"><li>After you finish using the returned `Index` pointer, remember to delete this pointer to release the corresponding memory.</li><li><code>devices</code> must be valid, non-duplicated device IDs, and the maximum number is 64.</li><li><code>index</code> must be a valid CPU `Index` pointer.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table22143401019"></a>
<table><tbody><tr id="row122113471017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p0263414104"><a name="p0263414104"></a><a name="p0263414104"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12818145232014"><a name="p12818145232014"></a><a name="p12818145232014"></a><code>faiss::Index *index_cpu_to_ascend(std::vector&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></p>
</td>
</tr>
<tr id="row1629341102"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212340103"><a name="p1212340103"></a><a name="p1212340103"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p82153419105"><a name="p82153419105"></a><a name="p82153419105"></a>Copies retrieval `Index` resources on the CPU and creates a retrieval `Index` on Ascend.</p>
</td>
</tr>
<tr id="row10211342101"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p182234111012"><a name="p182234111012"></a><a name="p182234111012"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p122133411107"><a name="p122133411107"></a><a name="p122133411107"></a><code>std::vector&lt;int&gt; devices</code>: Device IDs to configure on the NPU.</p>
<p id="p02113412104"><a name="p02113412104"></a><a name="p02113412104"></a><code>const faiss::Index *index</code>: Retrieval `Index` resources on the CPU.</p>
<p id="p521734121010"><a name="p521734121010"></a><a name="p521734121010"></a><code>const AscendClonerOptions *options = nullptr</code>: `AscendClonerOptions` resource to configure.</p>
</td>
</tr>
<tr id="row20253411010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p12734121017"><a name="p12734121017"></a><a name="p12734121017"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p22203431014"><a name="p22203431014"></a><a name="p22203431014"></a>None</p>
</td>
</tr>
<tr id="row15210342102"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p12283431010"><a name="p12283431010"></a><a name="p12283431010"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132173418105"><a name="p132173418105"></a><a name="p132173418105"></a>A retrieval `Index` on Ascend.</p>
</td>
</tr>
<tr id="row112143411101"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p5210340101"><a name="p5210340101"></a><a name="p5210340101"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1030011345514"></a><a name="ul1030011345514"></a><ul id="ul1030011345514"><li>After you finish using the returned `Index` pointer, remember to delete this pointer to release the corresponding memory.</li><li><code>devices</code> must be valid, non-duplicated device IDs, and the maximum number is 64.</li><li><code>index</code> must be a valid CPU `Index` pointer.</li></ul>
</td>
</tr>
</tbody>
</table>

## `index_int8_ascend_to_cpu`<a name="en-us_TOPIC_0000001506414761"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p01708458179"><a name="p01708458179"></a><a name="p01708458179"></a><code>faiss::Index *index_int8_ascend_to_cpu(const AscendIndexInt8 *index);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Copies INT8 retrieval `Index` resources on Ascend and creates a retrieval `Index` on the CPU.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p21724117137"><a name="p21724117137"></a><a name="p21724117137"></a><code>const AscendIndexInt8 *index</code>: `Index` resource on Ascend.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1246923411510"><a name="p1246923411510"></a><a name="p1246923411510"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>A retrieval `Index` on the CPU.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul20237164255510"></a><a name="ul20237164255510"></a><ul id="ul20237164255510"><li>After you finish using the returned `Index` pointer, remember to delete this pointer to release the corresponding memory.</li><li><code>index</code> must be a valid `AscendIndexInt8` pointer.</li></ul>
</td>
</tr>
</tbody>
</table>

## `index_int8_cpu_to_ascend`<a name="en-us_TOPIC_0000001456375248"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19640556122110"><a name="p19640556122110"></a><a name="p19640556122110"></a><code>AscendIndexInt8 *index_int8_cpu_to_ascend(std::initializer_list&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p894441552217"><a name="p894441552217"></a><a name="p894441552217"></a>Copies retrieval `Index` resources on the CPU and creates an INT8 retrieval `Index` on Ascend.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p21724117137"><a name="p21724117137"></a><a name="p21724117137"></a><code>std::initializer_list&lt;int&gt; devices</code>: Device IDs to configure on the NPU.</p>
<p id="p142883052016"><a name="p142883052016"></a><a name="p142883052016"></a><code>const faiss::Index *index</code>: Retrieval `Index` resources on the CPU.</p>
<p id="p15735115410199"><a name="p15735115410199"></a><a name="p15735115410199"></a><code>const AscendClonerOptions *options = nullptr</code>: `AscendClonerOptions` resource to configure.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1246923411510"><a name="p1246923411510"></a><a name="p1246923411510"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>An INT8 retrieval `Index` on Ascend.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul142782285619"></a><a name="ul142782285619"></a><ul id="ul142782285619"><li>After you finish using the returned `Index` pointer, remember to delete this pointer to release the corresponding memory.</li><li><code>devices</code> must be valid, non-duplicated device IDs, and the maximum number is 64.</li><li><code>index</code> must be a valid CPU `Index` pointer.</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table161071151116"></a>
<table><tbody><tr id="row0610181121116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p166107115116"><a name="p166107115116"></a><a name="p166107115116"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p561010115113"><a name="p561010115113"></a><a name="p561010115113"></a><code>AscendIndexInt8 *index_int8_cpu_to_ascend(std::vector&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></p>
</td>
</tr>
<tr id="row1161011116113"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p461011116112"><a name="p461011116112"></a><a name="p461011116112"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p6610181101117"><a name="p6610181101117"></a><a name="p6610181101117"></a>Copies retrieval `Index` resources on the CPU and creates an INT8 retrieval `Index` on Ascend.</p>
</td>
</tr>
<tr id="row166109141110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p14610131191111"><a name="p14610131191111"></a><a name="p14610131191111"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p36107118118"><a name="p36107118118"></a><a name="p36107118118"></a><code>std::vector&lt;int&gt; devices</code>: Device IDs to configure on the NPU.</p>
<p id="p1561010110117"><a name="p1561010110117"></a><a name="p1561010110117"></a><code>const faiss::Index *index</code>: Retrieval `Index` resources on the CPU.</p>
<p id="p561031161112"><a name="p561031161112"></a><a name="p561031161112"></a><code>const AscendClonerOptions *options = nullptr</code>: `AscendClonerOptions` resource to configure.</p>
</td>
</tr>
<tr id="row1461012111111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p36107113117"><a name="p36107113117"></a><a name="p36107113117"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p161071131117"><a name="p161071131117"></a><a name="p161071131117"></a>None</p>
</td>
</tr>
<tr id="row14611161171112"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p106113111111"><a name="p106113111111"></a><a name="p106113111111"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p16111161110"><a name="p16111161110"></a><a name="p16111161110"></a>An INT8 retrieval `Index` on Ascend.</p>
</td>
</tr>
<tr id="row146118119111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p5611101151114"><a name="p5611101151114"></a><a name="p5611101151114"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul20972759205619"></a><a name="ul20972759205619"></a><ul id="ul20972759205619"><li>After you finish using the returned `Index` pointer, remember to delete this pointer to release the corresponding memory.</li><li><code>devices</code> must be valid, non-duplicated device IDs, and the maximum number is 64.</li><li><code>index</code> must be a valid CPU `Index` pointer.</li></ul>
</td>
</tr>
</tbody>
</table>
