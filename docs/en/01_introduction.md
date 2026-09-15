# Introduction<a name="en-us_TOPIC_0000001668092436"></a>

## Index SDK Introduction

**Product Background<a name="section10853119102920"></a>**

With the development of artificial intelligence technologies in recent years, advanced algorithm models can effectively extract feature representations from unstructured data such as images, text, and speech. These feature representations are structured as vector features. In practical application scenarios, quickly and accurately finding vectors similar to a query vector has become an important requirement for various intelligent applications. This requires an efficient vector feature-based retrieval system, one of whose core components is an efficient retrieval engine.

Against this backdrop, an efficient vector feature retrieval engine has been implemented on the Huawei Ascend platform using Index SDK. You can build application-oriented retrieval systems on top of this engine.

**Product Definition<a name="section0743249122915"></a>**

FeatureRetrieval is a heterogeneous retrieval acceleration framework for Ascend NPUs developed based on Faiss. It provides high-performance retrieval for massive amounts of data in high-dimensional spaces. It is developed in C++ in a style consistent with Faiss, combined with TBE operators, and supports the ARM and x86_64 platforms. FeatureRetrieval supports two types of retrieval libraries: <b>small-library search (brute-force search)</b> and <b>large-library search (approximate nearest neighbor search)</b>. Small libraries typically contain 300,000 to 1 million entries, while large libraries can contain tens or even hundreds of millions of entries. The supported feature vector dimensions range from 64 to 512, depending on the algorithm.

- <b>Small-library search (brute-force search)</b> mainly implements brute-force search algorithms such as Flat, SQ, and INT8. It performs an exhaustive search of the feature vectors in the base library and returns the TopK results sorted by distance.
    - The INT8 algorithm performs brute-force search based on feature quantization and is therefore also called "int8flat" (for example, the operator generation script `int8flat_generate_model.py`).
    - The SQ algorithm performs quantization internally. Because it uses 8-bit integers for quantization, it is also called "SQ8" (for example, `sq8_generate_model.py`).

- <b>Large-library search (approximate nearest neighbor search)</b> implements the IVFSQ algorithm on the Ascend platform based on the Faiss feature retrieval framework and the IVF approach. Here, IVF differs from the traditional "inverted index." Its basic idea is to cluster the features first and then narrow the retrieval range using the cluster centers. This approach trades accuracy for performance.

The underlying implementation of each algorithm uses TBE operators accelerated on the Ascend platform.

In addition, FeatureRetrieval supports attribute-filtered search and multi-index batch search.

- **Attribute-filtered search** allows you to add temporal, spatial, and other attribute tags when adding vector data to the base library. During retrieval, you can specify attribute conditions so that only base-library data meeting the conditions is searched, enabling precise filtering.
- **Multi-index batch search** allows you to use multiple indexes to partition the data into separate libraries and retrieve data from multiple base libraries at once through a unified interface.

## Software Architecture<a name="en-us_TOPIC_0000001698142841"></a>

The Index SDK software architecture is shown in [Figure 1](#fig883164172512). The key modules in the architecture are described in [Table 1](#table3548152713258).

**Figure 1** Software architecture<a id="fig883164172512"></a>
![](figures/software-architecture.png "Software architecture")

**Table 1** Index SDK module introduction<a id="table3548152713258"></a>

| Module | Description |
| ------ | ----------- |
| Index SDK API layer | Provides Faiss-compatible C++ interfaces. Upper-layer applications can implement feature ingestion, query, deletion, and training functions. |
| Algorithm logic layer | Implements the logical flow of retrieval algorithms. The currently supported algorithms mainly include brute-force search, approximate nearest neighbor search, and attribute-filtering algorithms. |
| Operator layer | Provides acceleration operators for retrieval algorithms on the Ascend platform, including distance computation operators, TopK sorting operators, and attribute-filtering mask operators. |

## Getting Started<a name="en-us_TOPIC_0000001698062121"></a>

Recommended course: [Index SDK Feature Retrieval Beginner's Course](https://www.hiascend.com/edu/growth/details/310d161ab02c45958f9bc3d8fbbec51e)

**Usage Process<a name="section57646464414"></a>**

As shown in [Figure 2](#fig15421350143413), using Index SDK for feature retrieval can be divided into the following steps.

**Figure 2** Index SDK usage process<a id="fig15421350143413"></a>

![](figures/en-us_image_0000002148233552.png)

1. Install and deploy.

   1. Learn about the hardware form factors and operating systems supported by Index SDK. See "[Supported hardware and operating systems](#supported-hardware-and-operating-systems)".
   2. Learn about the installation of the required dependencies. See "[Dependency installation](./04_installation_guide.md#dependency-installation)".
   3. Learn about and complete the installation and deployment of Index SDK. See "[Offline installation](./04_installation_guide.md#offline-installation)".

2. Determine the retrieval type and algorithm.

   Learn about the retrieval types supported by Index SDK and the algorithms included in each retrieval type, including the use cases for each algorithm, the operators that need to be generated, and sample descriptions. Analyze your actual service requirements and determine the retrieval type and algorithm you need to use. See "[Algorithm introduction](./05_user_guide.md#algorithm-introduction)".

3. Generate operators.

   Generate the operators required by the algorithm. See "[Generating operators](./05_user_guide.md#generating-operators)".

4. Call the APIs to implement the algorithm and obtain the retrieval results. See "[API reference](./api/README.md)".

**Usage Notes<a name="section46981890503"></a>**

- The current FeatureRetrieval implementation in Index SDK is developed and adapted based on Ascend AI Processors and the open-source Faiss similarity search framework. Compatibility with any other hardware or heterogeneous computing platforms is outside the scope of this document and the product.
- FeatureRetrieval is deployed using AscendCL interfaces. Therefore, `aclInit` has already been called internally, and you do not need to call it again.
- The maximum capacity supported by a single index depends on the device-side memory size of the specific Ascend AI Processor. The service side needs to plan the number of indexes according to actual requirements to prevent memory overrun. You are advised to create fewer than 10,000 indexes in a single call to the `add` API. If more than 10,000 indexes are created, significant memory fragmentation may occur, which may cause the capacity of the `add` operation to be smaller than expected.

## Supported Hardware and Operating Systems<a name="en-us_TOPIC_0000001649663880"></a>

<table>
<tr>
<th>Product Series</th>
<th>Product Model</th>
<th>OS Versions (64-Bit Only)</th>
</tr>
<tr>
<td rowspan="5"><term>Atlas inference products</term></td>
<td>Atlas 300I Pro inference card</td>
<td><li>CentOS 7.6</li><li>openEuler 20.03</li><li>
openEuler 22.03</li><li>openEuler 24.03</li><li>Ubuntu 18.04</li><li>Ubuntu 20.04</li><li>EulerOS 2.12</li><li>EulerOS 2.15</li><li>KylinOS V10 SP3 2403</li><li>KylinOS V11</li><li>CTyunOS 23.01</li><li>UOS V20</li></td>
</tr>
<tr>
<td>Atlas 300V video analysis card</td>
<td><li>CentOS 7.6</li><li>openEuler 20.03</li><li>openEuler 22.03</li><li>Ubuntu 18.04</li><li>Ubuntu 20.04</li><li>EulerOS 2.12</li><li>UOS V20</li></td>
</tr>
<tr>
<td>Atlas 300V Pro video analysis card</td>
<td><li>CentOS 7.6</li><li>openEuler 20.03</li><li>openEuler 22.03</li><li>Ubuntu 18.04</li><li>Ubuntu 20.04</li><li>EulerOS 2.12</li><li>CTyunOS 23.01</li><li>UOS V20</li></td>
</tr>
<tr>
<td>Atlas 300I Duo inference card</td>
<td><li>CentOS 7.6</li><li>Ubuntu 18.04</li><li>Ubuntu 20.04</li><li>EulerOS 2.12</li><li>EulerOS 2.15</li><li>KylinOS V10 SP3 2403</li><li>KylinOS V11</li><li>openEuler 24.03</li><li>CTyunOS 23.01</li><li>UOS V20</li><li>UOS V25</li></td>
</tr>
<tr>
<td>Atlas 200I SoC A1 core board</td>
<td><li>CentOS 7.6</li><li>openEuler 20.03</li><li>EulerOS 2.12</li></td>
</tr>
<tr>
</tr>
<tr>
<td><term>Atlas A2 inference products</term>
<br>Note: <term>Atlas A2 inference products</term> support the AscendIndexFlat, AscendIndexInt8Flat, AscendIndexTS-FlatIP, AscendIndexTS-Int8Cos, AscendIndexIVFFlat, and AscendIndexIVFRaBitQ algorithms.</td>
<td>Atlas 800I A2 inference server</td>
<td><li>CentOS 7.6</li><li>openEuler 20.03</li><li>openEuler 22.03</li><li>openEuler 24.03</li><li>Ubuntu 18.04</li><li>Ubuntu 20.04</li><li>Ubuntu 24.04</li><li>EulerOS 2.12</li><li>EulerOS 2.15</li><li>UOS V20</li><li>UOS V25</li><li>KylinOS V10 SP3</li><li>KylinOS V11</li><li>BC-Linux_21.10 U4</li></td>
</tr>
<tr>
<td><term>Atlas A3 inference products</term><br>Note: The currently supported algorithms are AscendIndexFlat, AscendIndexTS-FlatIP, AscendIndexTS-Int8Cos, AscendIndexIVFFlat, and AscendIndexIVFRaBitQ.</td>
<td>Atlas 800I A3 SuperPoD server</td>
<td><li>Ubuntu 18.04</li><li>CUlinux 3.0</li><li>KylinOS V10 SP3 2403</li><li>KylinOS V11</li><li>CTyunOS 4</li><li>UOS V25</li></td>
</tr>
</table>
