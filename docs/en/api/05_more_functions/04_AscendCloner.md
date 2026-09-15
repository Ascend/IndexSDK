# `AscendCloner`<a name="en-us_TOPIC_0000001506334577"></a>

## Function Description<a name="en-us_TOPIC_0000001456375412"></a>

Index SDK provides an operation that copies retrieval `Index` resources on the NPU to Faiss on the CPU side. The copy process happens in memory. Data loaded in the original NPU `Index` is copied into CPU-side memory, which makes it convenient for users to run retrieval with the same base library on the CPU.

> [!NOTE]
> Some versions of Faiss provide a method for persisting an in-memory `Index` to disk, that is, saving in-memory data to a local drive. When you use Index SDK and Faiss to process sensitive data, pay special attention to the corresponding access control and encryption protection.

## `index_ascend_to_cpu`<a name="en-us_TOPIC_0000001506334821"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>faiss::Index *index_ascend_to_cpu(const faiss::Index *ascend_index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies retrieval <code>Index</code> resources on Ascend and creates a retrieval <code>Index</code> on the CPU.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const faiss::Index *ascend_index</code>: <code>Index</code> resource on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">A retrieval <code>Index</code> on the CPU.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">After you finish using this <code>index</code>, remember to <code>delete</code> this pointer to release the corresponding memory.</td></tr>
</tbody></table>

## `index_cpu_to_ascend`<a name="en-us_TOPIC_0000001456695032"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>faiss::Index *index_cpu_to_ascend(std::initializer_list&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies retrieval <code>Index</code> resources on the CPU and creates a retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device IDs to configure on the NPU.<br><code>const faiss::Index *index</code>: Retrieval <code>Index</code> resources on the CPU.<br><code>const AscendClonerOptions *options = nullptr</code>: <code>AscendClonerOptions</code> resource to configure.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">A retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">After you finish using this <code>index</code>, remember to <code>delete</code> this pointer to release the corresponding memory. <code>devices</code> must be valid, non-duplicated device IDs, and the maximum number is 64. <code>index</code> must be a valid CPU <code>Index</code> pointer.</td></tr>
</tbody></table>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>faiss::Index *index_cpu_to_ascend(std::vector&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies retrieval <code>Index</code> resources on the CPU and creates a retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device IDs to configure on the NPU.<br><code>const faiss::Index *index</code>: Retrieval <code>Index</code> resources on the CPU.<br><code>const AscendClonerOptions *options = nullptr</code>: <code>AscendClonerOptions</code> resource to configure.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">A retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">After you finish using this <code>index</code>, remember to <code>delete</code> this pointer to release the corresponding memory. <code>devices</code> must be valid, non-duplicated device IDs, and the maximum number is 64. <code>index</code> must be a valid CPU <code>Index</code> pointer.</td></tr>
</tbody></table>

## `index_int8_ascend_to_cpu`<a name="en-us_TOPIC_0000001506414761"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>faiss::Index *index_int8_ascend_to_cpu(const AscendIndexInt8 *index);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies INT8 retrieval <code>Index</code> resources on Ascend and creates a retrieval <code>Index</code> on the CPU.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>const AscendIndexInt8 *index</code>: <code>Index</code> resource on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">A retrieval <code>Index</code> on the CPU.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">After you finish using this <code>index</code>, remember to <code>delete</code> this pointer to release the corresponding memory. <code>index</code> must be a valid <code>AscendIndexInt8</code> pointer.</td></tr>
</tbody></table>

## `index_int8_cpu_to_ascend`<a name="en-us_TOPIC_0000001456375248"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8 *index_int8_cpu_to_ascend(std::initializer_list&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies retrieval <code>Index</code> resources on the CPU and creates an INT8 retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices</code>: Device IDs to configure on the NPU.<br><code>const faiss::Index *index</code>: Retrieval <code>Index</code> resources on the CPU.<br><code>const AscendClonerOptions *options = nullptr</code>: <code>AscendClonerOptions</code> resource to configure.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">An INT8 retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">After you finish using this <code>index</code>, remember to <code>delete</code> this pointer to release the corresponding memory. <code>devices</code> must be valid, non-duplicated device IDs, and the maximum number is 64. <code>index</code> must be a valid CPU <code>Index</code> pointer.</td></tr>
</tbody></table>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexInt8 *index_int8_cpu_to_ascend(std::vector&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Copies retrieval <code>Index</code> resources on the CPU and creates an INT8 retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices</code>: Device IDs to configure on the NPU.<br><code>const faiss::Index *index</code>: Retrieval <code>Index</code> resources on the CPU.<br><code>const AscendClonerOptions *options = nullptr</code>: <code>AscendClonerOptions</code> resource to configure.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">An INT8 retrieval <code>Index</code> on Ascend.</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">After you finish using this <code>index</code>, remember to <code>delete</code> this pointer to release the corresponding memory. <code>devices</code> must be valid, non-duplicated device IDs, and the maximum number is 64. <code>index</code> must be a valid CPU <code>Index</code> pointer.</td></tr>
</tbody></table>
