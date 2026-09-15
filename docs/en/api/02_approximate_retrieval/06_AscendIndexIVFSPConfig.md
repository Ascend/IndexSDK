# `AscendIndexIVFSPConfig`<a id="en-us_TOPIC_0000001635696057"></a>

`AscendIndexIVFSP` requires the corresponding `AscendIndexIVFSPConfig` to initialize the corresponding resources.

**Common Parameters<a name="section17656114673616"></a>**

<table><tbody>
<tr><td width="210" align="center" valign="middle">Parameter</td><td valign="middle">Data Type</td><td valign="middle">Parameter Description</td></tr>
<tr><td width="210" align="center" valign="middle">handleBatch</td><td valign="middle">int</td><td valign="middle">Number of candidate buckets submitted for computation each time during search. The default value is 64.</td></tr>
<tr><td width="210" align="center" valign="middle">nprobe</td><td valign="middle">int</td><td valign="middle">Total number of candidate buckets used during search. The default value is 64.</td></tr>
<tr><td width="210" align="center" valign="middle">searchListSize</td><td valign="middle">int</td><td valign="middle">Maximum number of samples in each bucket submitted for computation each time during search. The default value is 32768. If a bucket is too large, the program automatically splits the bucket into multiple operator submissions according to <code>searchListSize</code> to compute distances.</td></tr>
</tbody></table>

**API Description<a name="section74781713710"></a>**

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline AscendIndexIVFSPConfig();</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Default constructor. The default <code>devices</code> value is <code>{0}</code>, so the 0th Ascend AI Processor is used for computation. The default <code>resources</code> value is 128 MB.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<a name="table121971648373"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline explicit AscendIndexIVFSPConfig(std::initializer_list&lt;int&gt; devices, int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSPConfig</code>. It creates an <code>AscendIndexIVFSPConfig</code> and specifies the device IDs on the device side and the resource pool size.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::initializer_list&lt;int&gt; devices:</code> Device-side device IDs.<br><code>int64_t resources:</code> Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>IVF_SP_DEFAULT_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.<br><code>uint32_t blockSize:</code> Preallocated memory block size, in bytes. The default value is <code>DEFAULT_BLOCK_SIZE</code> in the header file.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs. Currently, only one NPU device is supported. The configured <code>resources</code> value must not exceed 10 \* 1024 MB, that is, 10 \* 1024 \* 1024 \* 1024 bytes.</td></tr>
</tbody></table>

<a name="table56061252785"></a>

<table><tbody>
<tr><td width="210" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>inline explicit AscendIndexIVFSPConfig(std::vector&lt;int&gt; devices, int64_t resources = IVF_SP_DEFAULT_MEM, uint32_t blockSize = DEFAULT_BLOCK_SIZE);</code></strong></td></tr>
<tr><td width="210" align="center" valign="middle">Description</td><td valign="middle">Constructor for <code>AscendIndexIVFSPConfig</code>. It creates an <code>AscendIndexIVFSPConfig</code> and specifies the device IDs on the device side and the resource pool size.</td></tr>
<tr><td width="210" align="center" valign="middle">Input</td><td valign="middle"><code>std::vector&lt;int&gt; devices:</code> Device-side device IDs.<br><code>int64_t resources:</code> Preallocated memory pool size on the device side, in bytes. This is the memory space used to store intermediate results during computation. It helps avoid performance fluctuations caused by dynamic memory allocation during computation. The default value is <code>IVF_SP_DEFAULT_MEM</code> in the header file. This parameter is determined by the base vector set size and the search batch count. When the base vector set size is greater than or equal to 10 million and the batch count is greater than or equal to 16, you are advised to set it to 1024 MB.<br><code>uint32_t blockSize:</code> Preallocated memory block size, in bytes. The default value is <code>DEFAULT_BLOCK_SIZE</code> in the header file.</td></tr>
<tr><td width="210" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="210" align="center" valign="middle">Constraints</td><td valign="middle"><code>devices</code> must be valid, unique device IDs. Currently, only one NPU device is supported. The configured <code>resources</code> value must not exceed 10 \* 1024 MB, that is, 10 \* 1024 \* 1024 \* 1024 bytes.</td></tr>
</tbody></table>
