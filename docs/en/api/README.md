# API Reference<a name="ZH-CN_TOPIC_0000001456534960"></a>

<table><tbody>
<tr><td width="280" align="center" valign="middle"><strong>Category</strong></td><td align="center" valign="middle"><strong>Link</strong></td></tr>
<tr><td width="280" align="center" valign="middle">Full retrieval</td><td valign="middle"><a href="./full_retrieval.md">full_retrieval</a></td></tr>
<tr><td width="280" align="center" valign="middle">Approximate retrieval</td><td valign="middle"><a href="./approximate_retrieval.md">approximate_retrieval</a></td></tr>
<tr><td width="280" align="center" valign="middle">Attribute filtering retrieval</td><td valign="middle"><a href="./attribute_filtering-based_retrieval.md">attribute_filtering-based_retrieval</a></td></tr>
<tr><td width="280" align="center" valign="middle">Multi-index batch retrieval</td><td valign="middle"><a href="./multi-index_batch_retrieval.md">multi-index_batch_retrieval</a></td></tr>
<tr><td width="280" align="center" valign="middle">Other functions</td><td valign="middle"><a href="./more_functions.md">more_functions</a></td></tr>
<tr><td width="280" align="center" valign="middle">Unused APIs</td><td valign="middle"><a href="./unused_apis.md">unused_apis</a></td></tr>
<tr><td width="280" align="center" valign="middle">API return value reference</td><td valign="middle"><a href="./return_code_reference.md">return_code_reference</a></td></tr>
</tbody></table>

## API Changes<a name="ZH-CN_TOPIC_0000001691057326"></a>

This section describes API changes, including additions, modifications, deletions, and retirement notices. API changes reflect only code-level changes. They do not include improvements to the document itself, such as language, format, or links.

- Added: Indicates an API added in this version.
- Modified: Indicates that the API changed compared with the previous version.
- Deleted: Indicates that the API was deleted in this version.
- Retirement notice: Indicates that the API stops evolving starting from the version in which the retirement notice is issued, and it is retired and removed one year later.

<table><tbody>
<tr><td align="center" valign="middle"><strong>Class Name/API Prototype</strong></td><td width="180" align="center" valign="middle"><strong>Change Type</strong></td><td align="center" valign="middle"><strong>Change Description</strong></td><td width="90" align="center" valign="middle"><strong>Version</strong></td></tr>
<tr><td valign="middle"><code>AscendIndexIVFConfig</code> / <code>AscendIndexIVFPQConfig</code></td><td width="180" align="center" valign="middle">Modified</td><td valign="middle">Move large-nlist training fields (<code>trainSamplesPerList</code>, <code>maxTrainSamples</code>, <code>pqNiter</code>, <code>useDistributedCoarse</code>) from <code>AscendIndexIVFConfig</code> to the end of <code>AscendIndexIVFPQConfig</code>, restoring derived IVF config POD layout (IVFFlat / IVFSQ / IVFRaBitQ). Fixes #133.</td><td width="90" align="center" valign="middle">26.2.0</td></tr>
<tr><td valign="middle">The <a href="./full_retrieval.md#init">Init</a> <code>AscendIndexCluster</code></td><td width="180" align="center" valign="middle">Modified</td><td valign="middle">The <code>resourceSize</code> variable in the <code>Init</code> API of the <code>AscendIndexCluster</code> algorithm uses the default value 128 MB.</td><td width="90" align="center" valign="middle">6.0.RC2</td></tr>
<tr><td valign="middle">Constructor of <code>AscendIndexBinaryFlat</code></td><td width="180" align="center" valign="middle">Modified</td><td valign="middle">The <code>AscendIndexBinaryFlat</code> constructor adds the <code>usedFloat</code> parameter, which improves the performance of the retrieval mode that stores binary features and uses float features for retrieval, namely, the <code>search</code> API.</td><td width="90" align="center" valign="middle">6.0.RC2</td></tr>
<tr><td valign="middle">The <a href="./approximate_retrieval.md#search">search</a> <code>AscendIndexBinaryFlat</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle"><code>AscendIndexBinaryFlat</code> adds support for the retrieval mode in which binary features are stored and float features are used for retrieval.</td><td width="90" align="center" valign="middle">6.0.RC2</td></tr>
<tr><td valign="middle"><a href="./full_retrieval.md#ascendindexint8flatconfig">AscendIndexInt8FlatConfig</a> <code>AscendIndexInt8Flat</code> (Table 2)</td><td width="180" align="center" valign="middle">Modified</td><td valign="middle">The value of <code>resourceSize</code> cannot exceed 16 \* 1024 MB (16 \* 1024 \* 1024 \* 1024 bytes).</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle"><a href="./full_retrieval.md#ascendindexint8flatconfig">AscendIndexInt8FlatConfig</a> <code>AscendIndexInt8Flat</code> (Table 3)</td><td width="180" align="center" valign="middle">Modified</td><td valign="middle">The value of <code>resourceSize</code> cannot exceed 16 \* 1024 MB (16 \* 1024 \* 1024 \* 1024 bytes).</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle">The <a href="./attribute_filtering-based_retrieval.md#init">Init</a> <code>AscendIndexTS</code></td><td width="180" align="center" valign="middle">Modified</td><td valign="middle">Changes the constraints on the <code>maxFeatureRowCount</code> parameter.</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle">The <a href="./full_retrieval.md#setpagesize">setPageSize</a> <code>AscendIndexInt8Flat</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">Sets the number of consecutive base-library blocks that <code>AscendIndexInt8Flat</code> computes in a single <code>search</code> call.</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle"><a href="./attribute_filtering-based_retrieval.md#initwithextraval">InitWithExtraVal</a> <code>AscendIndexTS</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">Initialization function for an instance with extra attributes.</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle"><a href="./attribute_filtering-based_retrieval.md#addwithextraval">AddWithExtraVal</a> <code>AscendIndexTS</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">API for adding features with additional attributes.</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle"><a href="./attribute_filtering-based_retrieval.md#getbasebyrangewithextraval">GetBaseByRangeWithExtraVal</a> <code>AscendIndexTS</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">Queries the base library with additional attributes by range.</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle"><a href="./attribute_filtering-based_retrieval.md#getextravalattrbylabel">GetExtraValAttrByLabel</a> <code>AscendIndexTS</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">Obtains the attributes of the specified label feature.</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle"><a href="./attribute_filtering-based_retrieval.md#extravalattr">ExtraValAttr</a> <code>AscendIndexTS</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">Additional attribute information.</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle"><a href="./attribute_filtering-based_retrieval.md#extravalfilter">ExtraValFilter</a> <code>AscendIndexTS</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">Additional attribute filter.</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle">The <a href="./approximate_retrieval.md#setremovefast">setRemoveFast</a> <code>AscendIndexBinaryFlat</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">Sets whether <code>AscendIndexBinaryFlat</code> quickly deletes vectors from the base library.</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle"><a href="./approximate_retrieval.md#ascendindexvstar">AscendIndexVStar</a></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">Adds the new <code>AscendIndexVStar</code> algorithm.</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle"><a href="./approximate_retrieval.md#ascendindexgreat">AscendIndexGreat</a></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">Adds the new <code>AscendIndexGreat</code> algorithm.</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle">The <a href="./approximate_retrieval.md#setsearchparams">setSearchParams</a> <code>AscendIndexIVFSQT</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">Sets the parameters that affect retrieval accuracy and performance.</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle">The <a href="./approximate_retrieval.md#setnumprobes">setNumProbes</a> <code>AscendIndexIVFSQT</code></td><td width="180" align="center" valign="middle">Retirement notice</td><td valign="middle">Expected to be deprecated in September 2025. Use <code>setSearchParams</code> instead.</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle">The <a href="./approximate_retrieval.md#updatetparams">updateTParams</a> <code>AscendIndexIVFSQT</code></td><td width="180" align="center" valign="middle">Retirement notice</td><td valign="middle">Expected to be deprecated in September 2025. Use <code>setSearchParams</code> instead.</td><td width="90" align="center" valign="middle">6.0.RC3</td></tr>
<tr><td valign="middle">The <a href="./attribute_filtering-based_retrieval.md#setsavehostmemory">SetSaveHostMemory</a> <code>AscendIndexTS</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">Sets the API for using the host-memory-saving mode.</td><td width="90" align="center" valign="middle">6.0.0</td></tr>
<tr><td valign="middle">The <a href="./full_retrieval.md#add">add</a> <code>AscendIndex</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">The Flat algorithm newly supports ingesting FP16 data into the base library.</td><td width="90" align="center" valign="middle">6.0.0</td></tr>
<tr><td valign="middle">The <a href="./full_retrieval.md#add_with_ids">add_with_ids</a> <code>AscendIndex</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">The Flat algorithm newly supports ingesting FP16 data into the base library with IDs.</td><td width="90" align="center" valign="middle">6.0.0</td></tr>
<tr><td valign="middle">The <a href="./full_retrieval.md#search">search</a> <code>AscendIndex</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">The Flat algorithm newly supports FP16 retrieval.</td><td width="90" align="center" valign="middle">6.0.0</td></tr>
<tr><td valign="middle">The <a href="./full_retrieval.md#search_with_masks">search_with_masks</a> <code>AscendIndexFlat</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">The Flat algorithm newly supports FP16 retrieval with masks.</td><td width="90" align="center" valign="middle">6.0.0</td></tr>
<tr><td valign="middle">The <a href="./approximate_retrieval.md#ascendindexivfsp">AscendIndexIVFSP</a> <code>AscendIndexIVFSP</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">Constructor for the shared-codebook mode.</td><td width="90" align="center" valign="middle">6.0.0</td></tr>
<tr><td valign="middle">The <a href="./approximate_retrieval.md#savealldata">saveAllData</a> <code>AscendIndexIVFSP</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">Stores IVFSP data in memory.</td><td width="90" align="center" valign="middle">6.0.0</td></tr>
<tr><td valign="middle">The <a href="./approximate_retrieval.md#loadalldata-api">loadAllData</a> <code>AscendIndexIVFSP</code></td><td width="180" align="center" valign="middle">Added</td><td valign="middle">Restores IVFSP from memory.</td><td width="90" align="center" valign="middle">6.0.0</td></tr>
</tbody></table>

## Calling Process and Inheritance Relations<a name="ZH-CN_TOPIC_0000001506615153"></a>

> [!NOTE]
> The C++ APIs of the Index SDK feature retrieval component follow the exception handling mechanism of the open-source Faiss API. Therefore, you must call them within `try`/`catch` blocks and handle exceptions there. For a detailed example, see the handling method in [Code Reference](../appendix.md#code-reference). This prevents program exits caused by exceptions during use.

The basic process for calling retrieval APIs is shown in [Figure 1 Basic process for calling retrieval APIs](#fig7270141171511).

**Figure 1** Basic process for calling retrieval APIs<a id="fig7270141171511"></a>

![](../figures/basic-process-for-calling-retrieval-apis.png "Basic process for calling retrieval APIs")

Feature retrieval inherits from `Index` in Faiss and supports multiple retrieval indexes. It provides APIs for building, querying, and deleting a base library. The inheritance relations among the objects are shown in [Figure 2 Inheritance relations among some AscendIndexConfig classes](#fig1028942114236) and [Figure 3 Inheritance relations among some AscendIndex classes](#fig13557318153512).

**Figure 2** Inheritance relations among some AscendIndexConfig classes<a id="fig1028942114236"></a>
![](../figures/inheritance-relations-among-some-ascendindexconfig-classes.png "Inheritance relations among some AscendIndexConfig classes")

**Figure 3** Inheritance relations among some AscendIndex classes<a id="fig13557318153512"></a>
![](../figures/inheritance-relations-among-some-ascendindex-classes.png "Inheritance relations among some AscendIndex classes")

> [!NOTE]
>
>- Because some feature retrieval inputs use pointer types, ensure that these pointers are valid. Otherwise, potential issues such as out-of-bounds reads or writes may occur during feature retrieval. In addition, feature retrieval helps the Ascend AI Processor perform vector retrieval computation, so you must ensure that the input Device ID is valid. Otherwise, the function may fail because the device connection fails.
>- [Faiss](https://github.com/facebookresearch/faiss) is a widely used vector retrieval acceleration library in the industry. To help ecosystem users quickly migrate vector retrieval clustering services from CPU/GPU platforms to the Ascend platform, the `AscendIndex` base class for many algorithms on the Ascend platform inherits from the `faiss::Index` class. The member variables `d` and `ntotal` in `faiss::Index` are public. When you use `AscendIndex` and its `AscendIndexInt8` subclasses, do not modify these public member variables directly.
>- This document no longer describes the member functions and variables of the base class `faiss::Index`.
>- For the `resourceSize` variable in the `Config` class, its purpose is to reserve memory for intermediate results during feature retrieval. The unit is bytes. You are advised to increase it when the base-library features are large, for example, more than 3 million, and the number of query requests is large. This helps avoid performance fluctuations caused by temporary memory allocation during retrieval. You are advised to set it to 1024 \* 1024 \* 1024 bytes.
>- When you create a new `Index`, the system compares it with the requested `resources`. If there is a difference, it releases the original memory resources and requests new ones according to the latest `Index` resources. You are advised to keep the overall `resources` value of the `Index` consistent.
>- You can set the operator execution timeout by setting the `MX_INDEX_SYNCHRONIZE_STREAM_TIME` environment variable. The unit is ms, and the value range is [60000, 1800000]. The default value is 300000.

## Header Files<a name="ZH-CN_TOPIC_0000001698168801"></a>

**Table 1** Header files

<table><tbody>
<tr><td align="center" valign="middle"><strong>Header File Name</strong></td><td align="center" valign="middle"><strong>Directory</strong></td><td align="center" valign="middle"><strong>Purpose</strong></td></tr>
<tr><td valign="middle">AscendCloner.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle">This header file provides the operation for copying retrieval <code>Index</code> resources from the NPU to Faiss on the CPU side. The copy process occurs in memory. Data loaded on the original NPU <code>Index</code> is copied to CPU-side memory, which makes it easy to use the same base library for retrieval on the CPU.</td></tr>
<tr><td valign="middle">AscendClonerOptions.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle">Provides configuration options.</td></tr>
<tr><td valign="middle">AscendIndex.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle"><code>AscendIndex</code> is the base class for most retrieval <code>Index</code> implementations in the feature retrieval component. It sits on top of Faiss and defines APIs for other indexes in feature retrieval.</td></tr>
<tr><td valign="middle">AscendIndexBinaryFlat.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle">This header file provides the Hamming-distance API class and defines the external Hamming-distance APIs.</td></tr>
<tr><td valign="middle">AscendIndexCluster.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle">External APIs of <code>AscendIndexCluster</code>.</td></tr>
<tr><td valign="middle">AscendIndexFlat.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle">This class mainly provides external APIs for Flat-FP16.</td></tr>
<tr><td valign="middle">AscendIndexIVF.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle"><code>AscendIndexIVF</code> is the base class for approximate retrieval and cannot be used directly.</td></tr>
<tr><td valign="middle">AscendIndexIVFSP.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle">Provides the external APIs for IVFSP. The core APIs include <code>add</code>, <code>add_with_ids</code>, <code>search</code>, and <code>search_with_filter</code>.</td></tr>
<tr><td valign="middle">AscendIndexIVFSQ.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle">External APIs for IVFSQ, including <code>train</code>, <code>copyto</code>, <code>copyfrom</code>, and the constructor.</td></tr>
<tr><td valign="middle">AscendIndexInt8.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle"><code>AscendIndex</code> is the base class for int8-type indexes in the feature retrieval component. It sits on top of Faiss and defines APIs for <code>IndexInt8Flat</code>.</td></tr>
<tr><td valign="middle">AscendIndexInt8Flat.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle">This class mainly provides external APIs for Flat-Int8.</td></tr>
<tr><td valign="middle">AscendIndexSQ.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle">External API definitions for SQ retrieval.</td></tr>
<tr><td valign="middle">AscendIndexTS.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle">External APIs for the spatiotemporal library, including the Hamming, <code>Int8Flat</code>, and <code>FP16Flat</code> algorithms.</td></tr>
<tr><td valign="middle">AscendMultiIndexSearch.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle">Provides the external APIs for multi-index retrieval.</td></tr>
<tr><td valign="middle">AscendNNInference.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle">External APIs for neural-network dimensionality reduction.</td></tr>
<tr><td valign="middle">AscendIndexIVFSQT.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/custom</td><td valign="middle">Contains the three-level IVFSQ retrieval algorithm with dimensionality reduction and fuzzy clustering. It reclusters each cluster. First it selects <code>nprobe</code> clusters based on the first-level clustering results. Then it selects <code>l2nprobe</code> clusters from all second-level clusters, and then it performs precise retrieval.</td></tr>
<tr><td valign="middle">IReduction.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/custom</td><td valign="middle"><code>IReduction</code> is the unified API for dimensionality reduction methods in the feature retrieval component. It currently supports the <code>PCAR</code> and <code>NN</code> dimensionality-reduction algorithms.</td></tr>
<tr><td valign="middle">Version.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/utils</td><td valign="middle">API for obtaining version information.</td></tr>
<tr><td valign="middle">ErrorCode.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/device/include</td><td valign="middle">Contains Index SDK error code information.</td></tr>
<tr><td valign="middle">IndexILFlat.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/device/include</td><td valign="middle">External API definition of <code>IndexILFlat</code>.</td></tr>
<tr><td valign="middle">IndexIL.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/device/include</td><td valign="middle">Base class for <code>IndexILFlat</code>. It cannot be used directly.</td></tr>
<tr><td valign="middle">AscendIndexGreat.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle">External API definition for Great retrieval.</td></tr>
<tr><td valign="middle">AscendIndexVStar.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle">External API definition for VStar retrieval.</td></tr>
<tr><td valign="middle">AscendIndexMixSearchParams.h</td><td valign="middle">${mxIndex_install_path}/mxIndex/include/faiss/ascend/</td><td valign="middle">External header file for the parameter structures required by VStar and Great retrieval.</td></tr>
</tbody></table>

> [!NOTE]
>
>${mxIndex_install_path} indicates the installation path of Index SDK.
