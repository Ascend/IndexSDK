# Usage Guide

## Generating Operators<a name="en-us_TOPIC_0000001985832236"></a>

After installing Index SDK, follow the instructions in this section to set the operator-related environment variables and generate the operators required by the algorithms.

> [!NOTE]
>
> - The AscendIndexFlat algorithm supports online operator conversion for L2 and IP distances. If the `MX_INDEX_USE_ONLINEOP` environment variable is set to `1` (run `export MX_INDEX_USE_ONLINEOP=1` to set it), the operator is converted and called online, and you do not need to generate the offline operators described in this section. When using online operators, you must explicitly call <b>\(void\)aclFinalize\(\)</b> at the end of the application. You must also include the header `acl/acl.h`.
> - For algorithms that do not support online operators, setting `MX_INDEX_USE_ONLINEOP=1` causes the program to fail.

**Procedure<a name="section13749124217108"></a>**

1. Enter the `mxIndex-{version}` installation directory. The directory and file names are described in [Table 1](#table81133951612).

    ```bash
    cd mxIndex-{version}
    ```

    **Table 1** Index SDK directories and files<a id="table81133951612"></a>

    | Directory or file | Description |
    | ----------------- | ----------- |
    | `device` | Contains the dynamic libraries and header files for the IndexIL algorithm. |
    | `filelist.txt` | Lists the files in the software package. |
    | `host` | Contains the retrieval dynamic libraries. When performing feature retrieval, link to the dynamic libraries in this directory. |
    | `include` | Contains the API header files. |
    | `lib` | Contains the retrieval dynamic libraries. Link these libraries to `host/lib`. |
    | `modelpath` | Directory for storing operator `.om` files. After compiling the operators, you can place the `.om` files in this directory. (Optional) |
    | `ops` | Contains the `custom_opp_<arch>.run` script for installing retrieval algorithm operators. |
    | `script` | Contains the `uninstall.sh` script for uninstalling the Index SDK package. |
    | `tools` | Contains the Python scripts for operator generation. |
    | `version.info` | Contains version-related information. |

2. Enter the `ops` directory. Before compiling the operators, set the `ASCEND_HOME`, `ASCEND_VERSION`, and `ASCEND_OPP_PATH` environment variables. Their default values are `~/Ascend`, `~/Ascend/ascend-toolkit/latest`, and `~/Ascend/ascend-toolkit/latest/opp`, respectively.

    ```bash
    export ASCEND_HOME=~/Ascend
    export ASCEND_VERSION=~/Ascend/ascend-toolkit/latest
    export ASCEND_OPP_PATH=~/Ascend/ascend-toolkit/latest/opp
    ```

    - `ASCEND_HOME` indicates the file storage path after the CANN toolkit is installed.
    - `ASCEND_VERSION` indicates the Ascend version currently in use. If the ATC tool is installed in `/usr/local/Ascend/ascend-toolkit/latest`, you do not need to set `ASCEND_HOME` or `ASCEND_VERSION`.
    - `ASCEND_OPP_PATH` indicates the root directory of the operator library. You need write permission for this directory.

    > [!NOTE]
    > The `MAX_COMPILE_CORE_NUMBER` environment variable specifies the number of CPU cores available during graph compilation and is used at operator runtime. The current default value is `1`, so you do not need to set it.

3. Run the corresponding script according to the system architecture.

    - Arm architecture:

        ```bash
        ./custom_opp_aarch64.run
        ```

    - x86_64 architecture:

        ```bash
        ./custom_opp_x86_64.run
        ```

    You can also specify optional command-line options when running the script, as shown in [Table 2](#table38211859291).

    **Table 2** custom_opp_{arch}.run options<a id="table38211859291"></a>

    | Option | Description |
    | --------- | ----------- |
    | `--help \| -h` | Displays help information. |
    | `--info` | Displays package build information. |
    | `--list` | Displays the file list. |
    | `--check` | Checks package integrity. |
    | `--quiet \| -q` | Optional. Enables silent installation and reduces interactive output. |
    | `--nox11` | Deprecated interface with no effect. |
    | `--noexec` | Extracts the software package to the current directory without running the installation script. Use it with `--extract=<path>`, in the form `--noexec --extract=<path>`. |
    | `--extract=<path>` | Extracts the files in the software package to the specified directory. Can be used with `--noexec`. |
    | `--tar arg1 [arg2 ...]` | Runs the `tar` command on the software package, using the parameters following `tar` as command arguments. For example, `--tar xvf` extracts the contents of the `.run` installation package to the current directory. |

    > [!NOTE]
    > The following options are not displayed in the `--help` output. Do not use them directly.
    > - `--xwin`: Runs in xwin mode.
    > - `--phase2`: Requires the second step to be executed.

4. Enter the `tools` directory to generate the required operators. Before generating the operators, ensure that the relevant CANN dependencies are installed.
    - To generate only the operators required by the algorithms you use, first refer to the [Algorithm Introduction](#algorithm-introduction) section to determine which operators need to be generated, and then refer to the [Custom Operator Introduction](#custom-operator-introduction) section to generate the corresponding operators.
    - To generate operators for all algorithms in batch, use the method shown in [Table 3](#table03891576018).

        **Table 3** Batch generation of operators<a id="table03891576018"></a>

        | Usage | `python3 run_generate_model.py -m <mode> -t <npu_type> -p <pipeline> -pool <pool_size>` |
        | ----------- | ----------- |
        | Parameters | `<mode>`: Algorithm mode. `<mode>` supports `ALL` and one or more of `Flat`, `SQ8`, `IVFSQ8`, and `INT8`. Separate multiple values with commas, for example, `python3 run_generate_model.py -m Flat,IVFSQ8`. All algorithms are selected by default, so you can run `python3 run_generate_model.py` directly.<br>`<npu_type>`: Chip name. <li>For <term>Atlas inference products</term>, run `npu-smi info` on the server where the Ascend AI Processor is installed. Remove the last digit from the reported `Name` value. The remaining value is the `npu_type` value.</li><li>For Atlas 800I A2 Inference Servers, run `npu-smi info` on the server where the Ascend AI Processor is installed. The reported `Name` value is the `npu_type` value.</li><li>For Atlas 800I A3 Supernode Servers, run `npu-smi info -t board -i 0 -c 0` to obtain the `NPU Name` value. `910_` followed by the `NPU Name` value is the `npu_type` value.</li><br>`<pipeline>`: Specifies whether to use multi-threaded parallel pipelines to generate operator models. The default value is `true`. When set to `true`, the default `pool_size` is `32`.<br>`<pool_size>`: Process pool size for multi-process scheduling during batch operator generation.<br>`--help \| -h`: Displays help information. |
        | Description | <li>Running this command generates multiple sets of operator model files. Before running the command, modify the `para_table.xml` file in the current directory and enter the required parameters in the parameter table.</li><li>`1 ≤ pool_size ≤ 32`</li> |

        > [!NOTE]
        > The constraints in the operator generation table represent parameter combinations commonly used in service scenarios. If an exception occurs when you run the tool with other parameters, see the [CANN Ascend Tensor Compiler (ATC) User Guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/atctool/atlasatc_16_0001.html).

5. Prepare the operator model files.

    - You can configure the operator model file directory using the `MX_INDEX_MODELPATH` environment variable. This environment variable supports paths beginning with `~`, relative paths, and absolute paths. The path must not contain symbolic links. When this variable is used, the path is converted to an absolute path.

        ```bash
        mv op_models/* $PWD/../modelpath
        export MX_INDEX_MODELPATH=`realpath $PWD/../modelpath`
        ```

    - If you do not configure the directory using the environment variable, move the operator model files to the `modelpath` directory in the current directory.

    After generating the operators, store the relevant `.om` files securely and ensure that they are not tampered with.

    > [!NOTE]
    > If `Failed to import Python module` is reported during operator generation, see [The NumPy Data Type `np.float_` Has Been Removed](./07_faq.md#the-numpy-data-type-npfloat_-has-been-removed) for a solution.

## Algorithm Introduction<a name="en-us_TOPIC_0000001649848468"></a>

> [!NOTE]
> Standard deployment primarily uses AI CPUs. The recommended ratio of Ctrl CPUs to AI CPUs is as follows.
>
> - For <term>Atlas inference products</term>, set it to 1:7.
> For details about the configuration command, see the [npu-smi command](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/260RC1/re/npu/npusmi_053.html).

### Brute-force Search<a name="en-us_TOPIC_0000001698088061"></a>

**Brute-force Search Algorithm Introduction<a name="section46312418528"></a>**

Brute-force search refers to calculating the distance between the query vector and every vector in the base library and returning the TopK results with the smallest distances. Brute-force search does not perform any pruning or approximation. Therefore, it provides the highest retrieval accuracy, but the computational workload increases with the size of the base library. It is suitable for scenarios with strict accuracy requirements and moderate-sized base libraries.

| Algorithm (API Reference) | Algorithm Usage Scenario | Operators to Generate | Sample Link |
| -- | -- | -- | -- |
| [AscendIndexInt8Flat](./api/01_full_retrieval/06_AscendIndexInt8Flat.md#ascendindexint8flat) | <li>Feature type: int8</li><li>Feature dimension: 64, 128, 256, 384, 512, 768, 1024</li><li>Distance type: L2 and IP</li><li>Calculation precision: High</li><li>Device memory usage: Low</li><li>Applicable scenario: Brute-force search scenarios with high accuracy requirements</li> | <li>[INT8Flat](#int8flat)</li><li>[AICPU](#aicpu)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexInt8Flat.cpp">Link</a> |
| [AscendIndexFlat](./api/01_full_retrieval/08_AscendIndexFlat.md#ascendindexflat) | <li>Feature type: FP32, FP16</li><li>Feature dimension: 32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096</li><li>Distance type: L2 and IP</li><li>Calculation precision: High</li><li>Device memory usage: High</li><li>Applicable scenario: Brute-force search scenarios with high accuracy requirements. IP distance is recommended when `dim > 128`.</li> | <li>[Flat](#flat)</li><li>[AICPU](#aicpu)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexFlat.cpp">Link</a> |
| [AscendIndexSQ](./api/01_full_retrieval/11_AscendIndexSQ.md#ascendindexsq) | <li>Feature type: FP32</li><li>Feature dimension: 64, 128, 256, 384, 512, 768</li><li>Distance type: L2 and IP</li><li>Calculation precision: High</li><li>Device memory usage: Low (quantized to int8)</li><li>Applicable scenario: Brute-force search scenarios with relatively high accuracy requirements</li> | <li>[SQ8](#sq8)</li><li>[AICPU](#aicpu)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexSQ.cpp">Link</a> |
| [AscendIndexCluster](./api/01_full_retrieval/02_AscendIndexCluster.md#ascendindexcluster) | <li>Feature type: FP32</li><li>Feature dimension: 32, 64, 128, 256, 384, 512</li><li>Distance type: IP</li><li>Calculation precision: High</li><li>Device memory usage: Relatively high</li><li>Applicable scenario: Distance-only clustering scenarios</li><li>Supported only on <term>Atlas inference products</term></li> | <li>[Flat](#flat)</li><li>[AICPU](#aicpu)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexCluster.cpp">Link</a> |
| [IndexIL](./api/01_full_retrieval/13_IndexIL.md#indexil) | Runs on the Device. Installation and deployment are complex, so it is not currently recommended. | <li>[Flat](#flat)</li> | See [IndexILFlat](./api/01_full_retrieval/14_IndexILFlat.md#indexilflat) |
| [AscendIndexILFlat](./api/01_full_retrieval/15_AscendIndexILFlat.md#ascendindexilflat) | <li>Feature type: FP16, FP32</li><li>Feature dimension: 32, 64, 128, 256, 384, 512</li><li>Distance type: IP</li><li>Calculation precision: High</li><li>Device memory usage: Relatively high</li><li>Applicable scenario: Distance-only clustering scenarios</li><li>Supported only on <term>Atlas inference products</term></li> | <li>[Flat](#flat)</li><li>[AICPU](#aicpu)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/tree/master/IndexSDK">Link</a> |

### Approximate Nearest Neighbor Search<a name="en-us_TOPIC_0000001698168797"></a>

**Approximate Nearest Neighbor Search Algorithm Introduction<a name="section46312418528"></a>**

Approximate nearest neighbor search preprocesses or compresses the base library through methods such as clustering, quantization, and graph indexing. During retrieval, it calculates distances for only a subset of vectors, trading a small amount of accuracy for significant performance improvements and memory savings. It is suitable for scenarios involving base libraries with billions of vectors, high performance requirements, latency sensitivity, and tolerance for some accuracy loss.

| Algorithm (API Reference) | Algorithm Usage Scenario | Operators to Generate | Sample Link |
| -- | -- | -- | -- |
| [AscendIndexIVFSP](./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#ascendindexivfsp) | <li>Feature type: FP32</li><li>Feature dimension: 64, 128, 256, 512, 768</li><li>Distance type: L2</li><li>Calculation precision: Medium</li><li>Device memory usage: Low (compressed features)</li><li>Applicable scenario: Approximate search scenarios involving base libraries with billions of vectors, high performance requirements, and tolerance for some accuracy loss.</li><li>Supported only on <term>Atlas inference products</term></li> | <li>IVFSP service operator</li><li>IVFSP AICPU operator</li><li>IVFSP training operator (used only when a codebook file needs to be generated through training)</li><br>See [IVFSP](#ivfsp). | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFSP.cpp">Link</a> |
| [AscendIndexIVFSQ](./api/02_approximate_retrieval/07_AscendIndexIVFSQ.md#ascendindexivfsq) | <li>Feature type: FP32</li><li>Feature dimension: 64, 128, 256, 384, 512</li><li>Distance type: L2 and IP</li><li>Calculation precision: Medium</li><li>Device memory usage: Low (quantized to int8)</li><li>Applicable scenario: The IVFSQ algorithm provides a performance-accuracy tradeoff and is suitable for scenarios that tolerate some accuracy loss but have high performance requirements.</li> | <li>[IVFSQ8](#ivfsq8)</li><li>[AICPU](#aicpu)</li><li>[FlatAT](#flatat) (generate the FlatAT operator only when `useKmeansPP` is set to `true`)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFSQ.cpp">Link</a> |
| [AscendIndexIVFSQT](./api/02_approximate_retrieval/09_AscendIndexIVFSQT.md#ascendindexivfsqt) | <li>Feature type: FP32</li><li>Feature dimension: 256</li><li>Distance type: IP</li><li>Calculation precision: Medium</li><li>Device memory usage: Low (quantization and dimensionality reduction)</li><li>Applicable scenario: AscendIndexIVFSQT is a three-stage IVFSQ retrieval algorithm that includes dimensionality reduction. It is suitable for approximate search scenarios involving base libraries with billions of vectors, high performance requirements, and tolerance for some accuracy loss.</li> | <li>[IVFSQT](#ivfsqt)</li><li>[FlatAT](#flatat)</li><li>[AICPU](#aicpu)</li><li>[FlatInt8AT](#flatint8at) (required on <term>Atlas inference products</term>)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFSQT.cpp">Link</a> |
| [AscendIndexBinaryFlat](./api/02_approximate_retrieval/01_AscendIndexBinaryFlat.md#ascendindexbinaryflat) | <li>Feature type: uint8 binary features</li><li>Feature dimension: 256, 512, 1024</li><li>Distance type: Hamming and IP</li><li>Calculation precision: High</li><li>Device memory usage: Low</li><li>Applicable scenario: The AscendIndexBinaryFlat class inherits from the Faiss `IndexBinary` class and is used for binary feature retrieval. It is suitable for scenarios with low memory usage requirements and high performance requirements.</li><li>Supported only on <term>Atlas inference products</term></li> | <li>[BinaryFlat](#binaryflat)</li><li>[AICPU](#aicpu)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexBinaryFlat.cpp">Link</a> |
| [AscendIndexVStar](./api/02_approximate_retrieval/11_AscendIndexVStar.md#ascendindexvstar) | <li>Feature type: FP32</li><li>Feature dimension: 128, 256, 512, 1024</li><li>Distance type: L2</li><li>Calculation precision: Medium</li><li>Device memory usage: Low (compressed features)</li><li>Applicable scenario: Approximate search scenarios involving base libraries with tens of millions of vectors, high performance requirements, and tolerance for some accuracy loss.</li><li>Supported only on <term>Atlas inference products</term></li> | <li>VStar service operator</li><li>VStar AICPU operator</li><li>VStar training operator (used only when a codebook file needs to be generated through training)</li><br>See [VSTAR](#vstar). | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexVStar.cpp">Link</a> |
| [AscendIndexGreat](./api/02_approximate_retrieval/12_AscendIndexGreat.md#ascendindexgreat) | <li>Feature type: FP32</li><li>Feature dimension: 128, 256, 512, 1024</li><li>Distance type: L2</li><li>Calculation precision: Medium</li><li>Device memory usage: Low (compressed features)</li><li>Applicable scenario: Approximate search scenarios involving base libraries with tens of millions of vectors, high performance requirements, and tolerance for some accuracy loss.</li><li>Supported only on <term>Atlas inference products</term> (the operator needs to be generated only when `mode` is `AKMode`).</li> | <li>VStar service operator</li><li>VStar AICPU operator</li><li>VStar training operator (used only when a codebook file needs to be generated through training)</li><br>See [VSTAR](#vstar). | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexGreat.cpp">Link</a> |
| [AscendIndexIVFFlat](./api/02_approximate_retrieval/14_AscendIndexIVFFlat.md#ascendindexivfflat) | <li>Feature type: FP32</li><li>Feature dimension: 128</li><li>Distance type: IP</li><li>Calculation precision: Medium</li><li>Device memory usage: Medium</li><li>Applicable scenario: Approximate search scenarios involving base libraries with billions of vectors, high performance requirements, and tolerance for some accuracy loss.</li><li>Supported only on <term>Atlas A2 Inference products</term>, <term>Atlas A3 Inference products</term>, and <term>Ascend 950 products</term></li> | <li>[AICPU](#aicpu)</li><li>[IVFFLAT](#ivfflat)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFFlat.cpp">Link</a> |
| [AscendIndexIVFPQ](./api/02_approximate_retrieval/15_AscendIndexIVFPQ.md#ascendindexivfpq) | <li>Feature type: FP32</li><li>Feature dimension: 128</li><li>Distance type: L2</li><li>Calculation precision: Medium (approximate search)</li><li>Device memory usage: Low (vectors compressed using PQ encoding)</li><li>Applicable scenario: Approximate search scenarios involving base libraries with billions of vectors, high throughput and low latency requirements, and tolerance for some accuracy loss.</li><li>Supported only on <term>Ascend 950 products</term></li> | <li>[AICPU](#aicpu)</li><li>[IVFPQ](#ivfpq)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFPQ.cpp">Link</a> |
| [AscendIndexIVFRaBitQ](./api/02_approximate_retrieval/16_AscendIndexIVFRaBitQ.md#ascendindexivfrabitq) | <li>Feature type: FP32</li><li>Feature dimension: 128</li><li>Distance type: L2 and IP</li><li>Calculation precision: Medium</li><li>Device memory usage: Low (compressed features)</li><li>Applicable scenario: Approximate search scenarios involving base libraries with billions of vectors, high performance requirements, and tolerance for some accuracy loss.</li><li>Supported only on <term>Atlas A2 Inference products</term>, <term>Atlas A3 Inference products</term>, and <term>Ascend 950 products</term></li> | <li>[AICPU](#aicpu)</li><li>[IVFRaBitQ](#ivfrabitq)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFRabitQ.cpp">Link</a> |

### Attribute-Filtered Search<a name="en-us_TOPIC_0000001649689168"></a>

**Attribute-Filtered Search Algorithm Introduction<a name="section46312418528"></a>**

Attribute-filtered search refers to filtering based on business attributes (such as time, space, additional attributes, and custom attributes) on top of vector search. Distance calculation and ranking are performed only on vectors that meet the attribute conditions, enabling spatiotemporal joint search. It is suitable for scenarios that require both similarity and attribute constraints.

| Algorithm (API Reference) | Algorithm Usage Scenario | Operators to Generate | Sample Link |
| -- | -- | -- | -- |
| [AscendIndexTS](./api/03_attribute_filtering-based_retrieval/01_AscendIndexTS.md#ascendindexts) | <li>Feature type: uint8 binary features, int8, and FP32, depending on the specific algorithm</li><li>Feature dimension: depends on the specific algorithm</li><li>Distance type: Hamming, Cos, IP, and L2</li><li>Calculation precision: Relatively high</li><li>Device memory usage: Relatively high</li><li>Applicable scenario: Spatiotemporal database scenarios that require attribute filtering</li><li>Cos and IP are supported on <term>Atlas inference products</term>, <term>Atlas A2 Inference products</term>, and <term>Atlas A3 Inference products</term></li><li>Hamming distance is supported only on <term>Atlas inference products</term></li> | <li>[Mask](#mask)</li><li>[BinaryFlat](#binaryflat)</li><li>[Int8Flat](#int8flat)</li><li>[Flat](#flat)</li><li>[AICPU](#aicpu)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexTS.cpp">Link</a> |

### Multi-Index Batch Search<a name="en-us_TOPIC_0000001649848472"></a>

**Multi-Index Batch Search Introduction<a name="section46312418528"></a>**

Multi-index batch search allows multiple Index instances to be managed simultaneously on a single Device and performs search on multiple Index instances with a single call, reducing the number of interactions between the Host and Device and improving overall throughput for concurrent multi-index search.

| Interface (API Reference) | Usage Scenario | Algorithms That Can Use This Interface | Sample Link |
| -- | -- | -- | -- |
| [Search](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#search-faissindex) | Search on multiple Index instances on a single Device. | <li>[AscendIndexSQ](./api/01_full_retrieval/11_AscendIndexSQ.md#ascendindexsq)</li><li>[AscendIndexFlat](./api/01_full_retrieval/08_AscendIndexFlat.md#ascendindexflat)</li><li>[AscendIndexIVFSP](./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#ascendindexivfsp)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a> |
| [Search](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#search-ascendindex) | Search on multiple AscendIndex instances on a single Device. | <li>[AscendIndexSQ](./api/01_full_retrieval/11_AscendIndexSQ.md#ascendindexsq)</li><li>[AscendIndexFlat](./api/01_full_retrieval/08_AscendIndexFlat.md#ascendindexflat)</li><li>[AscendIndexIVFSP](./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#ascendindexivfsp)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a> |
| [Search](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#search-ascendindexint8) | Search on multiple AscendIndexInt8 instances on a single Device. | <li>[AscendIndexInt8Flat](./api/01_full_retrieval/06_AscendIndexInt8Flat.md#ascendindexint8flat)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a> |
| [SearchWithFilter](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilter-faissindex-single-filter) | Search on multiple Index instances with attribute filtering (single filter) on a single Device. | <li>[AscendIndexSQ](./api/01_full_retrieval/11_AscendIndexSQ.md#ascendindexsq)</li><li>[AscendIndexIVFSP](./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#ascendindexivfsp)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a> |
| [SearchWithFilter](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilter-ascendindex-single-filter) | Search on multiple AscendIndex instances with attribute filtering (single filter) on a single Device. | <li>[AscendIndexSQ](./api/01_full_retrieval/11_AscendIndexSQ.md#ascendindexsq)</li><li>[AscendIndexIVFSP](./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#ascendindexivfsp)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a> |
| [SearchWithFilter](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilter-faissindex-multiple-filters) | Search on multiple Index instances with attribute filtering (multiple filters) on a single Device. | <li>[AscendIndexSQ](./api/01_full_retrieval/11_AscendIndexSQ.md#ascendindexsq)</li><li>[AscendIndexIVFSP](./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#ascendindexivfsp)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a> |
| [SearchWithFilter](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilter-ascendindex-multiple-filters) | Search on multiple AscendIndex instances with attribute filtering (multiple filters) on a single Device. | <li>[AscendIndexSQ](./api/01_full_retrieval/11_AscendIndexSQ.md#ascendindexsq)</li><li>[AscendIndexIVFSP](./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#ascendindexivfsp)</li> | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a> |

### Other Functions<a name="en-us_TOPIC_0000001698088065"></a>

**Algorithm Introduction<a name="section46312418528"></a>**

| Algorithm (API Reference) | Algorithm Requirements (Performance and Scenario Differences) | Invocation Method | Sample Link |
| -- | -- | -- | -- |
| [IReduction](./api/05_more_functions/01_IReduction.md#ireduction) | IReduction is a unified interface for dimensionality reduction methods in the feature retrieval component. It currently supports two dimensionality reduction algorithms: **PCAR** and **NN**. | Initialize ReductionConfig, call CreateReduction to create the dimensionality reduction object, and then call `train` and `reduce`. | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIReduction.cpp">Link</a> |
| [AscendNNInference](./api/05_more_functions/02_AscendNNInference.md#ascendnninference) | Performs inference through a neural network. | Create an NN dimensionality reduction object using AscendNNInference, and then call `infer` for dimensionality reduction. | <a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIReduction.cpp">Link</a> |
| [AscendCloner](./api/05_more_functions/04_AscendCloner.md#ascendcloner) | Index SDK provides an operation for copying retrieval Index resources from the NPU to a CPU-side Faiss Index. The copy is performed in memory. The data loaded in the original NPU Index is copied to CPU memory, allowing you to perform retrieval on the CPU using the same base library. | `index_ascend_to_cpu` copies an AscendIndex to create a CPU-side Index, and `index_cpu_to_ascend` copies a CPU-side Index to create an AscendIndex. | None |

## Custom Operator Introduction<a name="en-us_TOPIC_0000001456854988"></a>

### Custom Operator Overview<a name="en-us_TOPIC_0000001456695000"></a>

The feature retrieval solution uses TIK operators to implement feature distance calculation logic. It includes the following custom operators.

- [Flat distance calculation operator](#flat): Computes the distance between feature base library data and the feature vector to be searched (L2/IP).
- [SQ8 distance calculation operator](#sq8): Computes the distance between SQ-quantized feature base library data and the unquantized feature vector to be searched (L2/IP).
- [IVFSQ8 operator](#ivfsq8): Provides the operators required by the IVFSQ8 algorithm.
- [INT8Flat distance calculation operator](#int8flat): Computes the distance between INT8-quantized feature base library data and the INT8-quantized feature vector to be searched (L2/COS).
- [IVFSQT operator](#ivfsqt): Provides the distance operators required by the first, second, and third stages of IVFSQT.
- [FlatAT operator](#flatat): Mainly used in IVF scenarios to reduce the time consumed by <code>train</code> and <code>add</code>. Here, <code>code_num</code> is equal to <code>nlist</code>.
- [FlatInt8AT operator](#flatint8at): Reduces the time consumed by <code>train</code>, <code>add</code>, and <code>update</code> in IVFSQT on <term>Atlas inference products</term>.
- [AICPU operator](#aicpu): Schedules the CPU on the Ascend AI Processor to perform sorting and other calculations, making full use of hardware performance.
- [BinaryFlat operator](#binaryflat): Provides the operators required by binary algorithms.
- [Mask operator](#mask): Provides the Mask operator required by the spatiotemporal database attribute filtering algorithm.
- [IVFSP operator](#ivfsp): Provides the service operator and AICPU operator required by the IVFSP algorithm, as well as the training operator required to generate the IVFSP codebook during training.
- [VStar operator](#vstar): Provides the service operator and AICPU operator required by the VStar algorithm.
- [IVFFLAT](#ivfflat): Provides the distance operators required by the first and second stages of IVFFLAT.
- [IVFPQ operator](#ivfpq): Provides the distance operators required by the first, second, and third stages of IVFPQ.
- [IVFRaBitQ operator](#ivfrabitq): Provides the operators required by IVFRaBitQ.

### Operator Generation Instructions<a name="en-us_TOPIC_0000001456695052"></a>

#### Flat<a name="en-us_TOPIC_0000001506495813"></a>

<a name="table3955133174816"></a>
<table><tbody><tr id="row3956113304810"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p1995613338481"><a name="p1995613338481"></a><a name="p1995613338481"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p151131757175117"><a name="p151131757175117"></a><a name="p151131757175117"></a>python3 flat_generate_model.py -d &lt;dim&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row1695612338483"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p395693310480"><a name="p395693310480"></a><a name="p395693310480"></a>Parameter Names</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p1242365818554"><a name="p1242365818554"></a><a name="p1242365818554"></a>&lt;dim&gt;: Feature vector dimension D. The default value is <code>512</code>.</p>
<p id="p19895115711230"><a name="p19895115711230"></a><a name="p19895115711230"></a>&lt;core_num&gt;: Number of AI Cores on the Ascend AI Processor. The default value is <code>8</code>. No additional configuration is required.</p>
<p id="p1489519612244"><a name="p1489519612244"></a><a name="p1489519612244"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.</p>
<p id="p15972135916413"><a name="p15972135916413"></a><a name="p15972135916413"></a>&lt;pool_size&gt;: Process pool size for multi-process scheduling during batch operator generation. The default value is <code>10</code>.</p>
<p id="p16833155612195"><a name="p16833155612195"></a><a name="p16833155612195"></a>&lt;npu_type&gt;: Hardware form factor. Currently, &lt;npu_type&gt; supports <term>Atlas inference products</term>, <term>Atlas A2 Inference products</term>, and <term>Atlas A3 Inference products</term>. The default value is <code>310P</code>.</p>
<a name="ul994471125418"></a><a name="ul994471125418"></a><ul id="ul994471125418"><li><span id="ph10941163375016"><a name="ph10941163375016"></a><a name="ph10941163375016"></a>For <span id="ph19941183375011"><a name="ph19941183375011"></a><a name="ph19941183375011"></a><term>Atlas inference products</term></span>, run <strong id="b7330834135115"><a name="b7330834135115"></a><a name="b7330834135115"></a>npu-smi info</strong> on the server where the Ascend AI Processor is installed. Remove the last digit from the reported <code>Name</code> value to obtain the <code>npu_type</code> value.</span></li><li>For <span id="ph299603920504"><a name="ph299603920504"></a><a name="ph299603920504"></a>Atlas 800I A2 Inference Servers</span>, <span id="ph18599444165014"><a name="ph18599444165014"></a><a name="ph18599444165014"></a>run <strong id="b18495139195117"><a name="b18495139195117"></a><a name="b18495139195117"></a>npu-smi info</strong> on the server where the Ascend AI Processor is installed. The reported <code>Name</code> value is the <code>npu_type</code> value.</span></li><li>For <span id="ph6488102065112"><a name="ph6488102065112"></a><a name="ph6488102065112"></a>Atlas 800I A3 Supernode Servers</span>, run <strong id="b1248815206511"><a name="b1248815206511"></a><a name="b1248815206511"></a>npu-smi info -t board -i 0 -c 0</strong> to obtain the <strong id="b144882206516"><a name="b144882206516"></a><a name="b144882206516"></a>NPU Name</strong> value. <code>910_</code> followed by the <strong id="b1648872017519"><a name="b1648872017519"></a><a name="b1648872017519"></a>NPU Name</strong> value is the <code>npu_type</code> value.</li></ul>
<p id="p952414873216"><a name="p952414873216"></a><a name="p952414873216"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row15956133317485"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p11956183311486"><a name="p11956183311486"></a><a name="p11956183311486"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p9956933114816"><a name="p9956933114816"></a><a name="p9956933114816"></a>Run this command to generate a set of distance calculation operator model files. Modify the command parameters as required. For example, to generate a 512-dimensional operator for <term>Atlas inference products</term>, run the following command: python3 flat_generate_model.py -d 512 -t 310P</p>
</td>
</tr>
<tr id="row3636101012016"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p176373101504"><a name="p176373101504"></a><a name="p176373101504"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul9805193810583"></a><a name="ul9805193810583"></a><ul id="ul9805193810583"><li>dim ∈ {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}</li><li>0 ≤ pool_size ≤ 32</li></ul>
</td>
</tr>
</tbody>
</table>

**Involved Algorithms<a name="section1467921619472"></a>**

- [AscendIndexFlat](#brute-force-search)
- [AscendIndexCluster](#brute-force-search)
- [IndexIL](#brute-force-search)
- [AscendIndexTS](#attribute-filtered-search)
- [Search (multi-index searches on a single device)](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#search-faissindex)
- [Search (multi-AscendIndex searches on a single device)](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#search-ascendindex)

#### SQ8<a name="en-us_TOPIC_0000001506614921"></a>

> [!NOTE]
> The main difference between INT8Flat and SQ8 is that INT8 is quantized externally, and the input features to the Index are of type INT8. SQ8 is quantized internally by the Index, and the input features to the Index are of type Float32.

<a name="table3955133174816"></a>
<table><tbody><tr id="row3956113304810"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p1995613338481"><a name="p1995613338481"></a><a name="p1995613338481"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p185151728356"><a name="p185151728356"></a><a name="p185151728356"></a>python3 sq8_generate_model.py -d &lt;dim&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row1695612338483"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p395693310480"><a name="p395693310480"></a><a name="p395693310480"></a>Parameter Names</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p142073616319"><a name="p142073616319"></a><a name="p142073616319"></a>&lt;dim&gt;: Feature vector dimension D. The default value is <code>128</code>.</p>
<p id="p76412712341"><a name="p76412712341"></a><a name="p76412712341"></a>&lt;core_num&gt;: Number of AI Cores on the Ascend AI Processor. The default value is <code>8</code>. If this parameter is not specified, it is configured according to &lt;npu_type&gt;. When npu_type is 310P, &lt;core_num&gt; is 8.</p>
<p id="p882511475345"><a name="p882511475345"></a><a name="p882511475345"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.</p>
<p id="p822871512241"><a name="p822871512241"></a><a name="p822871512241"></a>&lt;pool_size&gt;: Process pool size for multi-process scheduling during batch operator generation. The default value is <code>10</code>.</p>
<p id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;: Hardware form factor. Currently, &lt;npu_type&gt; supports <span id="ph19590185162111"><a name="ph19590185162111"></a><a name="ph19590185162111"></a><term>Atlas inference products</term></span>. The valid value is <code>310P</code>, and the default value is <code>310P</code>.</p>
<p id="p1692503012329"><a name="p1692503012329"></a><a name="p1692503012329"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row15956133317485"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p11956183311486"><a name="p11956183311486"></a><a name="p11956183311486"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p158811120735"><a name="p158811120735"></a><a name="p158811120735"></a>Run this command to generate a set of SQ8 distance calculation operator model files. Modify the command parameters as required. For example, to generate a 512-dimensional operator for <term>Atlas inference products</term>, run the following command: python3 sq8_generate_model.py -d 512 -t 310P</p>
</td>
</tr>
<tr id="row1080311205318"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p280411209314"><a name="p280411209314"></a><a name="p280411209314"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul1361419421018"></a><a name="ul1361419421018"></a><ul id="ul1361419421018"><li>dim ∈ {64, 128, 256, 384, 512, 768}</li><li>0 ≤ pool_size ≤ 32</li></ul>
</td>
</tr>
</tbody>
</table>

**Involved Algorithms<a name="section6413836184719"></a>**

- [AscendIndexSQ](#brute-force-search)
- [Search (multi-index searches on a single device)](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#search-faissindex)
- [Search (multi-AscendIndex searches on a single device)](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#search-ascendindex)
- [SearchWithFilter (FaissIndex single filter)](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilter-faissindex-single-filter)
- [SearchWithFilter (AscendIndex single filter)](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilter-ascendindex-single-filter)
- [SearchWithFilter (FaissIndex multiple filters)](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilter-faissindex-multiple-filters)
- [SearchWithFilter (AscendIndex multiple filters)](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilter-ascendindex-multiple-filters)

#### IVFSQ8<a name="en-us_TOPIC_0000001506614889"></a>

<a name="table3955133174816"></a>
<table><tbody><tr id="row3956113304810"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p1995613338481"><a name="p1995613338481"></a><a name="p1995613338481"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p151131757175117"><a name="p151131757175117"></a><a name="p151131757175117"></a>python3 ivfsq8_generate_model.py -d &lt;dim&gt; -c &lt;coarse_centroid_num&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row1695612338483"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p395693310480"><a name="p395693310480"></a><a name="p395693310480"></a>Parameter Names</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p114841371610"><a name="p114841371610"></a><a name="p114841371610"></a>&lt;dim&gt;: Feature vector dimension D. The default value is <code>128</code>.</p>
<p id="p1157915132617"><a name="p1157915132617"></a><a name="p1157915132617"></a>&lt;coarse_centroid_num&gt;: Number of L1 cluster centroids. The default value is <code>16384</code>.</p>
<p id="p45741834183717"><a name="p45741834183717"></a><a name="p45741834183717"></a>&lt;core_num&gt;: Number of AI Cores on the <span id="ph129021410310"><a name="ph129021410310"></a><a name="ph129021410310"></a>Ascend AI Processor</span>. The default value is <span class="parmvalue" id="parmvalue76481515195212"><a name="parmvalue76481515195212"></a><a name="parmvalue76481515195212"></a><code>8</code></span>. If this parameter is not specified, it is configured according to &lt;npu_type&gt;: when <code>npu_type</code> is set to 310P, &lt;core_num&gt; is set to 8.</p>
<p id="p14268135033720"><a name="p14268135033720"></a><a name="p14268135033720"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is <span class="parmvalue" id="parmvalue12408135012357"><a name="p12408135012357"></a><a name="p12408135012357"></a><code>0</code></span>. No additional configuration is required.</p>
<p id="p17428219182417"><a name="p17428219182417"></a><a name="p17428219182417"></a>&lt;pool_size&gt;: Process pool size for multi-process scheduling during batch operator generation. The default value is <span class="parmvalue" id="parmvalue771714711233"><a name="parmvalue771714711233"></a><a name="parmvalue771714711233"></a><code>10</code></span>.</p>
<p id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;: Hardware form factor. Currently, &lt;npu_type&gt; supports <span id="ph19590185162111"><a name="ph19590185162111"></a><a name="ph19590185162111"></a><term>Atlas inference products</term></span>. The valid value is <code>310P</code>, and the default value is <span class="parmvalue" id="parmvalue68401116171220"><a name="parmvalue68401116171220"></a><a name="p68401116171220"></a><code>310P</code></span>.</p>
<p id="p824473918329"><a name="p824473918329"></a><a name="p824473918329"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row15956133317485"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p11956183311486"><a name="p11956183311486"></a><a name="p11956183311486"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p1648417714614"><a name="p1648417714614"></a><a name="p1648417714614"></a>Run this command to generate a set of operator model files. Modify the command parameters as required. For example, to generate operators for <term>Atlas inference products</term> with a dimension of 512 and an <code>nlist</code> of 1024, run the following command: python3 ivfsq8_generate_model.py -d 512 -c 1024 -t 310P</p>
</td>
</tr>
<tr id="row2657434476"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p1565893415718"><a name="p1565893415718"></a><a name="p1565893415718"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul1186212111343"></a><a name="ul1186212111343"></a><ul id="ul1186212111343"><li>dim ∈ {64, 128, 256, 384, 512}</li><li>coarse centroid num ∈ {1024, 2048, 4096, 8192, 16384, 32768}</li><li>0 ≤ pool_size ≤ 32</li></ul>
</td>
</tr>
</tbody>
</table>

**Involved Algorithms<a name="section14565105918474"></a>**

[AscendIndexIVFSQ](#approximate-nearest-neighbor-search)

#### INT8Flat<a name="en-us_TOPIC_0000001456695008"></a>

> [!NOTE]
> The main difference between INT8Flat and SQ8 is that INT8 is quantized externally, and the input features of the Index are of the INT8 type. SQ8 is quantized internally by the Index, and the input features of the Index are of the Float32 type.

<a name="table3955133174816"></a>
<table><tbody><tr id="row3956113304810"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p1995613338481"><a name="p1995613338481"></a><a name="p1995613338481"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p126011312484"><a name="p126011312484"></a><a name="p126011312484"></a>python3 int8flat_generate_model.py -d &lt;dim&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt; -code &lt;code_num&gt;</p>
</td>
</tr>
<tr id="row1695612338483"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p395693310480"><a name="p395693310480"></a><a name="p395693310480"></a>Parameter Names</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p114841371610"><a name="p114841371610"></a><a name="p114841371610"></a>&lt;dim&gt;: Feature vector dimension D. The default value is <code>512</code>.</p>
<p id="p76412712341"><a name="p76412712341"></a><a name="p76412712341"></a>&lt;core_num&gt;: Number of AI Cores on the <span id="ph129021410310"><a name="ph129021410310"></a><a name="ph129021410310"></a>Ascend AI Processor</span>. The default value is <span class="parmvalue" id="parmvalue1790915218539"><a name="parmvalue1790915218539"></a><a name="parmvalue1790915218539"></a><code>8</code></span>. No additional configuration is required.</p>
<p id="p1489519612244"><a name="p1489519612244"></a><a name="p1489519612244"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is <span class="parmvalue" id="parmvalue12408135012357"><a name="parmvalue12408135012357"></a><a name="parmvalue12408135012357"></a><code>0</code></span>. No additional configuration is required.</p>
<p id="p67082264240"><a name="p67082264240"></a><a name="p67082264240"></a>&lt;pool_size&gt;: Process pool size for multi-process scheduling during batch operator generation. The default value is <span class="parmvalue" id="parmvalue771714711233"><a name="parmvalue771714711233"></a><a name="parmvalue771714711233"></a><code>10</code></span>.</p>
<p id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;: Hardware form factor. Currently, &lt;npu_type&gt; supports <term>Atlas A2 Inference products</term> and <term>Atlas A3 Inference products</term>. The default value is <code>310P</code>.<a name="ul10641721165519"></a><a name="ul10641721165519"></a><ul id="ul10641721165519"><li><span id="ph10941163375016"><a name="ph10941163375016"></a><a name="ph10941163375016"></a>For <span id="ph19941183375011"><a name="ph19941183375011"></a><a name="ph19941183375011"></a><term>Atlas inference products</term></span>, run <strong id="b7330834135115"><a name="b7330834135115"></a><a name="b7330834135115"></a>npu-smi info</strong> on the server where the Ascend AI Processor is installed. Remove the last digit from the reported "Name" value to obtain the <code>npu_type</code> value.</span></li><li>For <span id="ph299603920504"><a name="ph299603920504"></a><a name="ph299603920504"></a>Atlas 800I A2 Inference Servers</span>, <span id="ph18599444165014"><a name="ph18599444165014"></a><a name="ph18599444165014"></a>run <strong id="b18495139195117"><a name="b18495139195117"></a><a name="b18495139195117"></a>npu-smi info</strong> on the server where the Ascend AI Processor is installed. The reported "Name" value is the <code>npu_type</code> value.</span></li></ul></p>

<p id="p6501256288"><a name="p6501256288"></a><a name="p6501256288"></a>&lt;code_num&gt;: Base library block size used when the operator is called. The default value is <code>262144</code>. If this parameter is not specified, operators for all <code>code_num</code> values are generated by default.</p>
<p id="p11599745183215"><a name="p11599745183215"></a><a name="p11599745183215"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row15956133317485"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p11956183311486"><a name="p11956183311486"></a><a name="p11956183311486"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p1648417714614"><a name="p1648417714614"></a><a name="p1648417714614"></a>Run this command to generate a set of operator model files. Modify the command parameters as required. For example, to generate operators for <term>Atlas inference products</term> with a dimension of 512, run the following command: python3 int8flat_generate_model.py -d 512 -t 310P</p>
</td>
</tr>
<tr id="row13262151218181"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p1552013434810"><a name="p1552013434810"></a><a name="p1552013434810"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul13923029345"></a><a name="ul13923029345"></a><ul id="ul13923029345"><li>dim ∈ {64, 128, 256, 384, 512, 768, 1024}</li><li>0 ≤ pool_size ≤ 32</li><li>code_num ∈ {16384, 32768, 65536, 131072, 262144}</li></ul>
</td>
</tr>
</tbody>
</table>

**Involved Algorithms<a name="section3261111214818"></a>**

- [AscendIndexInt8Flat](#brute-force-search)
- [AscendIndexTS](#attribute-filtered-search)
- [Search (multiple AscendIndexInt8 searches on a single Device)](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#search-ascendindexint8)

#### IVFSQT<a name="en-us_TOPIC_0000001506414677"></a>

> [!NOTE]
>
> To reduce the time consumed by <code>train</code> and <code>add</code>, you need to generate the FlatAT operator. The <code>dim</code> of Flat must be the same as the <code>dim_in</code> of IVFSQT, and the <code>code_num</code> of Flat must match the <code>coarse_centroid_num</code> of IVFSQT.

<a name="table3955133174816"></a>
<table><tbody><tr id="row3956113304810"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p1995613338481"><a name="p1995613338481"></a><a name="p1995613338481"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p151131757175117"><a name="p151131757175117"></a><a name="p151131757175117"></a>python3 ivfsqt_generate_model.py --cores &lt;core_num&gt; -d &lt;dim_in&gt; -r &lt;compress_ratio&gt; -c &lt;coarse_centroid_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row1695612338483"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p395693310480"><a name="p395693310480"></a><a name="p395693310480"></a>Parameter Names</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p114841371610"><a name="p114841371610"></a><a name="p114841371610"></a>&lt;dim_in&gt;: Input feature vector dimension. The default value is <code>256</code>.</p>
<p id="p334923265916"><a name="p334923265916"></a><a name="p334923265916"></a>&lt;compress_ratio&gt;: Ratio of input to output dimensions. The default value is <span class="parmvalue" id="parmvalue144721783518"><a name="parmvalue144721783518"></a><a name="parmvalue144721783518"></a><code>4</code></span>. Range: <code>compress_ratio</code> ≥ 1.</p>
<p id="p1157915132617"><a name="p1157915132617"></a><a name="p1157915132617"></a>&lt;coarse_centroid_num&gt;: Number of L1 cluster centroids. The default value is <code>16384</code>.</p>
<p id="p656513128471"><a name="p656513128471"></a><a name="p656513128471"></a>&lt;core_num&gt;: Number of AI Cores on the <span id="ph129021410310"><a name="ph129021410310"></a><a name="ph129021410310"></a>Ascend AI Processor</span>. The default value is <span class="parmvalue" id="parmvalue18901171585410"><a name="parmvalue18901171585410"></a><a name="parmvalue18901171585410"></a><code>8</code></span>. If this parameter is not specified, it is configured according to &lt;npu_type&gt;: when <code>npu_type</code> is set to 310P, &lt;core_num&gt; is set to 8.</p>
<p id="p1489519612244"><a name="p1489519612244"></a><a name="p1489519612244"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is <span class="parmvalue" id="parmvalue12408135012357"><a name="p12408135012357"></a><a name="p12408135012357"></a><code>0</code></span>. No additional configuration is required.</p>
<p id="p176142029102418"><a name="p176142029102418"></a><a name="p176142029102418"></a>&lt;pool_size&gt;: Process pool size for multi-process scheduling during batch operator generation. The default value is <span class="parmvalue" id="parmvalue1756444519167"><a name="p1756444519167"></a><a name="p1756444519167"></a><code>32</code></span>. Range: <code>1 ≤ pool_size ≤ 32</code>.</p>
<p id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;: Hardware form factor. Currently, &lt;npu_type&gt; supports <span id="ph19590185162111"><a name="ph19590185162111"></a><a name="ph19590185162111"></a><term>Atlas inference products</term></span>. The valid value is <code>310P</code>, and the default value is <span class="parmvalue" id="parmvalue68401116171220"><a name="parmvalue68401116171220"></a><a name="p68401116171220"></a><code>310P</code></span>.</p>
<p id="p581105217324"><a name="p581105217324"></a><a name="p581105217324"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row15956133317485"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p11956183311486"><a name="p11956183311486"></a><a name="p11956183311486"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p1648417714614"><a name="p1648417714614"></a><a name="p1648417714614"></a>Run this command to generate a set of operator model files. For example, to generate operators for <term>Atlas inference products</term> with an input dimension of 256, an output dimension of 64, and an <code>nlist</code> of 1024, run the following command: python3 ivfsqt_generate_model.py -d 256 -r 4 -c 1024 -t 310P</p>
</td>
</tr>
<tr id="row1329410259210"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p132941725162113"><a name="p132941725162113"></a><a name="p132941725162113"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul423417236171"></a><a name="ul423417236171"></a><ul id="ul423417236171"><li>&lt;dim_in&gt; ∈ {256}</li><li>&lt;compress_ratio&gt; ∈ {2, 4, 8}</li><li>&lt;coarse_centroid_num&gt; ∈ {1024, 2048, 4096, 8192, 16384, 32768}</li><li>&lt;dim_in&gt; must be divisible by &lt;compress_ratio&gt;.</li></ul>
</td>
</tr>
</tbody>
</table>

**Involved Algorithms<a name="section1931762794815"></a>**

[AscendIndexIVFSQT](#approximate-nearest-neighbor-search)

#### FlatAT<a name="en-us_TOPIC_0000001506414881"></a>

> [!NOTE]
> The current FlatAT operator is used together with IVF-type operators to accelerate the <code>add</code> and <code>train</code> processes of IVF-type operators. You cannot call the FlatAT operator directly. The current <code>add</code> and <code>train</code> acceleration feature is specified through AscendIndexIVFConfig.useKmeansPP in IVF. In this case, only training datasets with fewer than 7,000,000 vectors are supported.

<a name="table17415417319"></a>
<table><tbody><tr id="row124224153110"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p194211412319"><a name="p194211412319"></a><a name="p194211412319"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p1442184113118"><a name="p1442184113118"></a><a name="p1442184113118"></a>python3 flat_at_generate_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -c &lt;code_num&gt; -p &lt;process_id&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row11421741163119"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p1742941193110"><a name="p1742941193110"></a><a name="p1742941193110"></a>Parameter Names</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p2421641133115"><a name="p2421641133115"></a><a name="p2421641133115"></a>&lt;dim&gt;: Input feature vector dimension. The default value is <code>64</code>.</p>
<p id="p44274114316"><a name="p44274114316"></a><a name="p44274114316"></a>&lt;code_num&gt;: Number of base library features to compare with the input feature. The default value is <code>8192</code>.</p>
<p id="p16163133719589"><a name="p16163133719589"></a><a name="p16163133719589"></a>&lt;core_num&gt;: Number of AI Cores on the <span id="ph129021410310"><a name="ph129021410310"></a><a name="ph129021410310"></a>Ascend AI Processor</span>. The default value is <span class="parmvalue" id="parmvalue645218305563"><a name="parmvalue645218305563"></a><a name="parmvalue645218305563"></a><code>8</code></span>. If this parameter is not specified, it is configured according to &lt;npu_type&gt;: when <code>npu_type</code> is set to 310P, &lt;core_num&gt; is set to 8.</p>
<p id="p1222812464588"><a name="p1222812464588"></a><a name="p1222812464588"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is <span class="parmvalue" id="parmvalue19166152815386"><a name="parmvalue19166152815386"></a><a name="p19166152815386"></a><code>0</code></span>. No additional configuration is required.</p>
<p id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;: Hardware form factor. Currently, &lt;npu_type&gt; supports <span id="ph19590185162111"><a name="ph19590185162111"></a><a name="ph19590185162111"></a><term>Atlas inference products</term></span>. The valid value is <code>310P</code>, and the default value is <span class="parmvalue" id="parmvalue68401116171220"><a name="parmvalue68401116171220"></a><a name="p68401116171220"></a><code>310P</code></span>.</p>
<p id="p1989920599326"><a name="p1989920599326"></a><a name="p1989920599326"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row142104123115"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p124294113115"><a name="p124294113115"></a><a name="p124294113115"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p19612144442216"><a name="p19612144442216"></a><a name="p19612144442216"></a>Run this command to generate a set of operator model files. For example, to generate operators for <term>Atlas inference products</term> with a dimension of 256 and an <code>nlist</code> of 1024, run the following command: python3 flat_at_generate_model.py -d 256 -c 1024 -t 310P</p>
<p id="p4425415312"><a name="p4425415312"></a><a name="p4425415312"></a>The FlatAT operator is mainly used in IVF scenarios to reduce the time consumed by <code>train</code> and <code>add</code>.</p>
</td>
</tr>
<tr id="row1828715702415"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p12287757152416"><a name="p12287757152416"></a><a name="p12287757152416"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul49294250179"></a><a name="ul49294250179"></a><ul id="ul49294250179"><li>dim ∈ {64, 128, 256}</li><li>code_num ∈ {1024, 2048, 4096, 8192, 16384, 32768}</li></ul>
</td>
</tr>
</tbody>
</table>

**Involved Algorithms<a name="section019718356489"></a>**

- [AscendIndexIVFSQ](#approximate-nearest-neighbor-search)
- [AscendIndexIVFSQT](#approximate-nearest-neighbor-search)

#### FlatInt8AT<a name="en-us_TOPIC_0000001456694972"></a>

<a name="table17415417319"></a>
<table><tbody><tr id="row124224153110"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p194211412319"><a name="p194211412319"></a><a name="p194211412319"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p1442184113118"><a name="p1442184113118"></a><a name="p1442184113118"></a>python3 flat_at_int8_generate_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -c &lt;code_num&gt; -p &lt;process_id&gt; --soc-version &lt;soc_version&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row11421741163119"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p1742941193110"><a name="p1742941193110"></a><a name="p1742941193110"></a>Parameter Names</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p483612454218"><a name="p483612454218"></a><a name="p483612454218"></a>&lt;core_num&gt;: Number of AI Cores on the <span id="ph71911442141813"><a name="ph71911442141813"></a><a name="ph71911442141813"></a>Ascend AI Processor</span>. The default value is <span class="parmvalue" id="parmvalue11837165055814"><a name="parmvalue11837165055814"></a><a name="parmvalue11837165055814"></a><code>8</code></span>.</p>
<p id="p157945377424"><a name="p157945377424"></a><a name="p157945377424"></a>&lt;dim&gt;: Input feature vector dimension. The default value is <code>256</code>.</p>
<p id="p44274114316"><a name="p44274114316"></a><a name="p44274114316"></a>&lt;code_num&gt;: Number of base library features to compare with the input feature. The default value is <code>16384</code>.</p>
<p id="p1222812464588"><a name="p1222812464588"></a><a name="p1222812464588"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is <span class="parmvalue" id="parmvalue13397924123910"><a name="parmvalue13397924123910"></a><a name="p13397924123910"></a><code>0</code></span>. No additional configuration is required.</p>
<p id="p716454113415"><a name="p716454113415"></a><a name="p716454113415"></a>&lt;soc_version&gt;: Model of the <span id="ph129021410310"><a name="ph129021410310"></a><a name="ph129021410310"></a>Ascend AI Processor</span>. The default value is <span class="parmvalue" id="parmvalue198811757104210"><a name="parmvalue198811757104210"></a><a name="p198811757104210"></a><code>Ascend310P3</code></span>. No additional configuration is required.</p>
<p id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;: Hardware form factor. Currently, &lt;npu_type&gt; supports <span id="ph19590185162111"><a name="ph19590185162111"></a><a name="ph19590185162111"></a><term>Atlas inference products</term></span>. The default value is <span class="parmvalue" id="parmvalue68401116171220"><a name="p68401116171220"></a><a name="p68401116171220"></a><code>310P</code></span>. No additional configuration is required.</p>
<p id="p418616673318"><a name="p418616673318"></a><a name="p418616673318"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row142104123115"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p124294113115"><a name="p124294113115"></a><a name="p124294113115"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p19612144442216"><a name="p19612144442216"></a><a name="p19612144442216"></a>Run this command to generate a set of operator model files. For example, to generate operators for <term>Atlas inference products</term> with a dimension of 256 and an <code>nlist</code> of 1024, run the following command: python3 flat_at_int8_generate_model.py -d 256 -c 1024 -t 310P </p>
<p id="p137541630185615"><a name="p137541630185615"></a><a name="p137541630185615"></a>The FlatInt8AT operator optimizes the time consumed by <code>train</code>, <code>add</code>, and <code>update</code> in IVFSQT when used with <span id="ph9726111217394"><a name="ph9726111217394"></a><a name="ph9726111217394"></a>Atlas inference products</span>.</p>
</td>
</tr>
<tr id="row1828715702415"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p12287757152416"><a name="p12287757152416"></a><a name="p12287757152416"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul49294250179"></a><a name="ul49294250179"></a><ul id="ul49294250179"><li>dim ∈ {256}</li><li>code_num ∈ {1024, 2048, 4096, 8192, 16384, 32768}</li><li>soc_version ∈ {Ascend310P3}</li></ul>
</td>
</tr>
</tbody>
</table>

**Involved Algorithms<a name="section16686174317488"></a>**

[AscendIndexIVFSQT](#approximate-nearest-neighbor-search)

#### AICPU<a name="en-us_TOPIC_0000001506414793"></a>

<a name="table4331184817108"></a>
<table><tbody><tr id="row1433117485104"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p733211482108"><a name="p733211482108"></a><a name="p733211482108"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p1333211482107"><a name="p1333211482107"></a><a name="p1333211482107"></a>python3 aicpu_generate_model.py --cores &lt;core_num&gt; -p &lt;process_id&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row2033244801010"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p4332164821010"><a name="p4332164821010"></a><a name="p4332164821010"></a>Parameter Names</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p193321480106"><a name="p193321480106"></a><a name="p193321480106"></a>&lt;core_num&gt;: Number of AI Cores on the <span id="ph71911442141813"><a name="ph71911442141813"></a><a name="ph71911442141813"></a>Ascend AI Processor</span>. The default value is <span class="parmvalue" id="parmvalue4332204851012"><a name="parmvalue4332204851012"></a><a name="parmvalue4332204851012"></a>"2"</span>. (Reserved parameter, not currently used.)</p>
<p id="p43321548181012"><a name="p43321548181012"></a><a name="p43321548181012"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is <span class="parmvalue" id="parmvalue12408135012357"><a name="parmvalue12408135012357"></a><a name="ph12408135012357"></a>"0"</span>. No additional configuration is required.</p>
<p id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;: Hardware form factor. Currently, &lt;npu_type&gt; supports <span id="ph19590185162111"><a name="ph19590185162111"></a><a name="ph19590185162111"></a><term>Atlas inference products</term></span>, <span id="ph996833614580"><a name="ph996833614580"></a><a name="ph996833614580"></a><term>Atlas A2 Inference products</term>, <term>Atlas A3 Inference products</term></span>. The default value is <span class="parmvalue" id="parmvalue68401116171220"><a name="parmvalue68401116171220"></a><a name="parmvalue68401116171220"></a>"310P"</span>. If you cannot determine the exact npu_type, run the <strong id="b87057513481"><a name="b87057513481"></a><a name="b87057513481"></a>npu-smi info</strong> command on the server where the Ascend AI Processor is installed. Remove the last digit from the reported "Name" to obtain the value of npu_type. For <span id="ph12325145818223"><a name="ph12325145818223"></a><a name="ph12325145818223"></a>Atlas 800I A3 Supernode Servers</span>, you can run the <strong id="b10641459664"><a name="b10641459664"></a><a name="b10641459664"></a>npu-smi info -t board -i 0 -c 0</strong> command to obtain <strong id="b172691331852"><a name="b172691331852"></a><a name="b172691331852"></a>NPU Name</strong>. The value of npu_type is 910_ followed by <strong id="b223011104513"><a name="b223011104513"></a><a name="b223011104513"></a>NPU Name</strong>.</p>
<p id="p13676151710337"><a name="p13676151710337"></a><a name="p13676151710337"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row1333284817107"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p103329483101"><a name="p103329483101"></a><a name="p103329483101"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p1433264811011"><a name="p1433264811011"></a><a name="p1433264811011"></a>Running this command generates a set of operator model files. For example, to generate AICPU operators for Atlas inference products: python3 aicpu_generate_model.py -t 310P</p>
<p id="p7405349165018"><a name="p7405349165018"></a><a name="p7405349165018"></a>AICPU operator model files only need to be generated once, and operators for all algorithms are generated.</p>
</td>
</tr>
</tbody>
</table>

**Involved Algorithms<a name="section156851751144816"></a>**

- [AscendIndexInt8Flat](#brute-force-search)
- [AscendIndexFlat](#brute-force-search)
- [AscendIndexSQ](#brute-force-search)
- [AscendIndexCluster](#brute-force-search)
- [AscendIndexIVFSQ](#approximate-nearest-neighbor-search)
- [AscendIndexBinaryFlat](#approximate-nearest-neighbor-search)
- [AscendIndexTS](#attribute-filtered-search)
- [AscendIndexIVFSQT](#approximate-nearest-neighbor-search)
- [AscendIndexIVFFlat](#approximate-nearest-neighbor-search)
- [AscendIndexIVFPQ](#approximate-nearest-neighbor-search)
- [AscendIndexIVFRaBitQ](#approximate-nearest-neighbor-search)

#### BinaryFlat<a name="en-us_TOPIC_0000001506615001"></a>

<a name="table4331184817108"></a>
<table><tbody><tr id="row1433117485104"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p733211482108"><a name="p733211482108"></a><a name="p733211482108"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p1333211482107"><a name="p1333211482107"></a><a name="p1333211482107"></a>python3 binary_flat_generate_model.py -d &lt;dim&gt; -q &lt;query_type&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt;</p>
</td>
</tr>
<tr id="row2033244801010"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p4332164821010"><a name="p4332164821010"></a><a name="p4332164821010"></a>Parameter Names</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p193321480106"><a name="p193321480106"></a><a name="p193321480106"></a>&lt;dim&gt;: Dimension of the binary feature vector. dim ∈ { 256, 512, 1024 }. The default value is <span class="parmvalue" id="parmvalue12920913153"><a name="parmvalue12920913153"></a><a name="parmvalue12920913153"></a>"512"</span>.</p>
<p id="p1474218499117"><a name="p1474218499117"></a><a name="p1474218499117"></a>&lt;query_type&gt;: Search type. The default value is <span class="parmvalue" id="parmvalue12920913153"><a name="parmvalue12920913153"></a><a name="parmvalue12920913153"></a>"uint8"</span>. To improve the performance of the <a href="./api/02_approximate_retrieval/01_AscendIndexBinaryFlat.md#en-us_TOPIC_0000001456375288">search interface</a> of the AscendIndexBinaryFlat algorithm, set it to <span class="parmvalue" id="parmvalue10202131541419"><a name="parmvalue10202131541419"></a><a name="parmvalue10202131541419"></a>"float"</span>.</p>
<p id="p43321548181012"><a name="p43321548181012"></a><a name="p43321548181012"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is <span class="parmvalue" id="parmvalue12408135012357"><a name="parmvalue12408135012357"></a><a name="parmvalue12408135012357"></a>"0"</span>. No additional configuration is required.</p>
<p id="p670916785418"><a name="p670916785418"></a><a name="p670916785418"></a>&lt;pool_size&gt;: Process pool size for multi-process scheduling during batch operator generation. The default value is 16.</p>
<p id="p7767123203320"><a name="p7767123203320"></a><a name="p7767123203320"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row1333284817107"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p103329483101"><a name="p103329483101"></a><a name="p103329483101"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p1533224810102"><a name="p1533224810102"></a><a name="p1533224810102"></a>For example, to generate 256-dimensional uint8 operators for Atlas inference products: python3 binary_flat_generate_model.py -d 256.</p>
</td>
</tr>
</tbody>
</table>

**Involved Algorithms<a name="section6613359134811"></a>**

- [AscendIndexBinaryFlat](#approximate-nearest-neighbor-search)
- [AscendIndexTS](#attribute-filtered-search)

#### Mask<a name="en-us_TOPIC_0000001461181500"></a>

<a name="table4331184817108"></a>
<table><tbody><tr id="row1433117485104"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p733211482108"><a name="p733211482108"></a><a name="p733211482108"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p1197352833411"><a name="p1197352833411"></a><a name="p1197352833411"></a>python3 mask_generate_model.py -token &lt;max_token_cnt&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row2033244801010"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p4332164821010"><a name="p4332164821010"></a><a name="p4332164821010"></a>Parameter Names</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p16675175613311"><a name="p16675175613311"></a><a name="p16675175613311"></a>&lt;max_token_cnt&gt;: Maximum number of tokens for operator generation. The default value is 2500. The recommended range is [1, 300000].</p>
<p id="p1315743173515"><a name="p1315743173515"></a><a name="p1315743173515"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is <span class="parmvalue" id="parmvalue12408135012357"><a name="parmvalue12408135012357"></a><a name="parmvalue12408135012357"></a>"0"</span>. No additional configuration is required.</p>
<p id="p10793175223414"><a name="p10793175223414"></a><a name="p10793175223414"></a>&lt;pool_size&gt;: Process pool size for multi-process scheduling during batch operator generation. The default value is 16.</p>
<p id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;: Hardware form factor. Currently, &lt;npu_type&gt; supports <term>Atlas inference products</term>, <term>Atlas A2 Inference products</term>, and <term>Atlas A3 Inference products</term>. The default value is "310P".
<a name="ul994471125418"></a><a name="ul994471125418"></a><ul id="ul994471125418"><li><span id="ph10941163375016"><a name="ph10941163375016"></a><a name="ph10941163375016"></a>For <span id="ph19941183375011"><a name="ph19941183375011"></a><a name="ph19941183375011"></a>Atlas inference products</span>, run the <strong id="b7330834135115"><a name="b7330834135115"></a><a name="b7330834135115"></a>npu-smi info</strong> command on the server where the Ascend AI Processor is installed. Remove the last digit from the reported "Name" to obtain the value of npu_type.</span></li><li>For <span id="ph299603920504"><a name="ph299603920504"></a><a name="ph299603920504"></a>Atlas 800I A2 Inference Servers</span>, <span id="ph18599444165014"><a name="ph18599444165014"></a><a name="ph18599444165014"></a>run the <strong id="b18495139195117"><a name="b18495139195117"></a><a name="b18495139195117"></a>npu-smi info</strong> command on the server where the Ascend AI Processor is installed. The reported "Name" is the value of npu_type.</span></li><li>For <span id="ph6488102065112"><a name="ph6488102065112"></a><a name="ph6488102065112"></a>Atlas 800I A3 Supernode Servers</span>, run the <strong id="b1248815206511"><a name="b1248815206511"></a><a name="b1248815206511"></a>npu-smi info -t board -i 0 -c 0</strong> command to obtain <strong id="b144882206516"><a name="b144882206516"></a><a name="b144882206516"></a>NPU Name</strong>. The value of npu_type is 910_ followed by <strong id="b1648872017519"><a name="b1648872017519"></a><a name="b1648872017519"></a>NPU Name</strong>.</li></ul></p>
<p id="p688819307338"><a name="p688819307338"></a><a name="p688819307338"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row1333284817107"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p103329483101"><a name="p103329483101"></a><a name="p103329483101"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p1896717588333"><a name="p1896717588333"></a><a name="p1896717588333"></a>For example, to generate operators with 300000 tokens for Atlas inference products: python3 mask_generate_model.py -token 300000 -t 310P.</p>
</td>
</tr>
</tbody>
</table>

**Involved Algorithms<a name="section1345318864915"></a>**

[AscendIndexTS](#attribute-filtered-search)

#### IVFSP<a name="en-us_TOPIC_0000001635696757"></a>

IVFSP retrieval currently supports the "910B4" hardware form factor. It involves the generation of the following types of model files:

- ivfsp_generate_model.py: Generates IVFSP service operator model files. For details, see [IVFSP Service Operator Model File Generation](#section11272703813).
- ivfsp_aicpu_generate_model.py: Generates IVFSP AICPU operator model files. For details, see [IVFSP AICPU Operator Model File Generation](#section10476137113814).
- ivfsp_generate_pyacl_model.py: Generates the training operator model files required for IVFSP codebook training. For details, see [IVFSP Training Operator Model File Generation](#section51314823813).

**IVFSP Service Operator Model File Generation<a ID="section11272703813"></a>**

<a name="table4331184817108"></a>
<table><tbody><tr id="row1433117485104"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p733211482108"><a name="p733211482108"></a><a name="p733211482108"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p828815402278"><a name="p828815402278"></a><a name="p828815402278"></a>python3 ivfsp_generate_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -nonzero_num &lt;low_dim&gt; -nlist &lt;k&gt; -handle_batch &lt;handle_batch&gt; -code_num &lt;code_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt;</p>
</td>
</tr>
<tr id="row2033244801010"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p4332164821010"><a name="p4332164821010"></a><a name="p4332164821010"></a>Parameter Names</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p472394917248"><a name="p472394917248"></a><a name="p472394917248"></a>&lt;core_num&gt;: Number of AI Cores. The default value is "8". No additional configuration is required.</p>
<p id="p19724949202417"><a name="p19724949202417"></a><a name="p19724949202417"></a>&lt;dim&gt;: Feature vector dimension. The default value is "256".</p>
<p id="p472484911241"><a name="p472484911241"></a><a name="p472484911241"></a>&lt;low_dim&gt;: Number of non-zero dimensions after feature vector compression. The default value is "32".</p>
<p id="p1572494913240"><a name="p1572494913240"></a><a name="p1572494913240"></a>&lt;k&gt;: Number of cluster centroids. Keep this consistent with &lt;k&gt; in <a href="#section51314823813">IVFSP Training Operator Model File Generation</a>. The default value is "1024".</p>
<p id="p1972464912414"><a name="p1972464912414"></a><a name="p1972464912414"></a>&lt;handle_batch&gt;: Number of candidate buckets dispatched for calculation each time during retrieval. The default value is "32".</p>
<p id="p1272411499246"><a name="p1272411499246"></a><a name="p1272411499246"></a>&lt;code_num&gt;: Maximum number of samples per bucket dispatched for calculation each time during retrieval. If a bucket is too large, the program automatically splits it into multiple operator dispatches based on code_num to calculate distances. Keep this consistent with &lt;codebook_batch_size&gt; in <a href="#section51314823813">IVFSP Training Operator Model File Generation</a>. The default value is "32768".</p>
<p id="p5724134992412"><a name="p5724134992412"></a><a name="p5724134992412"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is <span class="parmvalue" id="parmvalue12408135012357"><a name="parmvalue12408135012357"></a><a name="parmvalue12408135012357"></a>"0"</span>. No additional configuration is required.</p>
<p id="p1626301617420"><a name="p1626301617420"></a><a name="p1626301617420"></a>&lt;pool_size&gt;: Process pool size for multi-process scheduling during batch operator generation. The default value is "16".</p>
<p id="p11852192143213"><a name="p11852192143213"></a><a name="p11852192143213"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row1333284817107"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p103329483101"><a name="p103329483101"></a><a name="p103329483101"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p3370256123818"><a name="p3370256123818"></a><a name="p3370256123818"></a>Running this command generates a set of AI Core operator model files for IVFSP retrieval. You need to modify the parameters in the command yourself. For example, to generate operators for Atlas inference products with a dimension of 256, a compressed dimension of 32, 1024 cluster centroids, 32 buckets, and 32768 samples: python3 ivfsp_generate_model.py -d 256 -nonzero_num 32 -nlist 1024 -handle_batch 32 -code_num 32768</p>
</td>
</tr>
<tr id="row1827720142259"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p327810141252"><a name="p327810141252"></a><a name="p327810141252"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul2098017146167"></a><a name="ul2098017146167"></a><ul id="ul2098017146167"><li>When dim ∈ {64, 128, 256}, k ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}; when dim ∈ {512, 768}, k ∈ {256, 512, 1024, 2048}.</li><li>low_dim must be a multiple of 16 and less than or equal to min(128, dim).</li><li>handle_batch must be a multiple of 16, and 16 ≤ handle_batch ≤ 240.</li><li>0 &lt; pool_size ≤ 32.</li></ul>
</td>
</tr>
</tbody>
</table>

**IVFSP AICPU Operator Model File Generation<a id="section10476137113814"></a>**

<a name="table1844216303913"></a>
<table><tbody><tr id="row124438353916"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p1944314323914"><a name="p1944314323914"></a><a name="p1944314323914"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p164439315396"><a name="p164439315396"></a><a name="p164439315396"></a>python3 ivfsp_aicpu_generate_model.py --cores &lt;core_num&gt; -p &lt;process_id&gt;</p>
</td>
</tr>
<tr id="row1344373183918"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p244314363910"><a name="p244314363910"></a><a name="p244314363910"></a>Parameter Names</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p11745153613262"><a name="p11745153613262"></a><a name="p11745153613262"></a>&lt;core_num&gt;: Number of AI Cores. The default value is "8". No additional configuration is required.</p>
<p id="p13745113617269"><a name="p13745113617269"></a><a name="p13745113617269"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is <span class="parmvalue" id="parmvalue1134717100258"><a name="parmvalue1134717100258"></a><a name="parmvalue1134717100258"></a>"0"</span>. No additional configuration is required.</p>
<p id="p31671221377"><a name="p31671221377"></a><a name="p31671221377"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row44439314393"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p1444316303912"><a name="p1444316303912"></a><a name="p1444316303912"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p144431434393"><a name="p144431434393"></a><a name="p144431434393"></a>Running this command generates a set of AICPU operator model files for IVFSP retrieval. For example, to generate AICPU operators for Atlas inference products: python3 ivfsp_aicpu_generate_model.py --cores 8.</p>
</td>
</tr>
</tbody>
</table>

**IVFSP Training Operator Model File Generation<a id="section51314823813"></a>**

<a name="table142311552394"></a>
<table><tbody><tr id="row12231105113915"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p62311050394"><a name="p62311050394"></a><a name="p62311050394"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p192321351392"><a name="p192321351392"></a><a name="p192321351392"></a>python3 ivfsp_generate_pyacl_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -nonzero_num &lt;low_dim&gt; -nlist &lt;k&gt; -batch_size &lt;batch_size&gt; -code_num &lt;codebook_batch_size&gt; -p &lt;process_id&gt;</p>
</td>
</tr>
<tr id="row723219523911"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p1023265163915"><a name="p1023265163915"></a><a name="p1023265163915"></a>Parameter Names</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p27921438102719"><a name="p27921438102719"></a><a name="p27921438102719"></a>&lt;core_num&gt;: Number of AI Cores. The default value is "8". No additional configuration is required.</p>
<p id="p19792438182711"><a name="p19792438182711"></a><a name="p19792438182711"></a>&lt;dim&gt;: Feature vector dimension. The default value is "256".</p>
<p id="p0792163819273"><a name="p0792163819273"></a><a name="p0792163819273"></a>&lt;low_dim&gt;: Number of non-zero dimensions after feature vector compression. The default value is "32".</p>
<p id="p20792133842718"><a name="p20792133842718"></a><a name="p20792133842718"></a>&lt;k&gt;: Number of cluster centroids. Keep this consistent with &lt;k&gt; in <a href="#section11272703813">IVFSP Service Operator Model File Generation</a>. The default value is "1024".</p>
<p id="p15792438122717"><a name="p15792438122717"></a><a name="p15792438122717"></a>&lt;batch_size&gt;: Batch size used during training. The default value is "32768".</p>
<p id="p87921838132719"><a name="p87921838132719"></a><a name="p87921838132719"></a>&lt;codebook_batch_size&gt;: Maximum number of samples used to operate on the codebook in each training operation. It must be a power of 2. Keep this consistent with &lt;code_num&gt; in <a href="#section11272703813">IVFSP Service Operator Model File Generation</a>. The default value is "32768".</p>
<p id="p17792838132718"><a name="p17792838132718"></a><a name="p17792838132718"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is <span class="parmvalue" id="parmvalue11499918132520"><a name="parmvalue11499918132520"></a><a name="parmvalue11499918132520"></a>"0"</span>. No additional configuration is required.</p>
<p id="p194632813370"><a name="p194632813370"></a><a name="p194632813370"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row182322051393"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p1223218512395"><a name="p1223218512395"></a><a name="p1223218512395"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p12232650398"><a name="p12232650398"></a><a name="p12232650398"></a>Running this command generates a set of operator model files for IVFSP retrieval. You need to modify the parameters in the command yourself. The generated IVFSP training operator model files are saved in the op_models_pyacl subdirectory of the current directory. For example, to generate operators for Atlas inference products with a dimension of 256, a compressed dimension of 32, 1024 cluster centroids for nlist, a query count of 32768, and a sample count of 32768: python3 ivfsp_generate_pyacl_model.py -d 256 -nonzero_num 32 -nlist 1024 -batch_size 32768 -code_num 32768</p>
</td>
</tr>
<tr id="row1265606112615"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p10656267261"><a name="p10656267261"></a><a name="p10656267261"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul18345712132414"></a><a name="ul18345712132414"></a><ul id="ul18345712132414"><li>When dim ∈ {64, 128, 256}, k ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}; when dim ∈ {512, 768}, k ∈ {256, 512, 1024, 2048}.</li><li>low_dim must be a multiple of 16 and less than or equal to min(128, dim).</li><li>batch_size must be a multiple of 16.</li><li>codebook_batch_size must be a multiple of 16.</li></ul>
</td>
</tr>
</tbody>
</table>

#### VSTAR<a name="en-us_TOPIC_0000002044867041"></a>

VSTAR retrieval currently supports only <term>Atlas inference products</term>. It involves generating the VSTAR service operator model file (`vstar_generate_models.py`). For details, see [VSTAR](#vstar).

The operator generation environment must be consistent with the codebook generation environment. For details, see [Overall Description](#overall-description).

**VSTAR Service Operator Model File Generation<a name="section11272703813"></a>**

<a name="table4331184817108"></a>
<table><tbody><tr id="row1433117485104"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p733211482108"><a name="p733211482108"></a><a name="p733211482108"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p828815402278"><a name="p828815402278"></a><a name="p828815402278"></a>python3 vstar_generate_models.py --dim &lt;dim&gt; --nlistL1 &lt;nlist1&gt;  --subDimL1 &lt;sub_dim1&gt;  --nProbeL1 &lt;nprobe1&gt; --nProbeL2 &lt;nprobe2&gt; --segmentNumL3 &lt;segment&gt; --pool &lt;pool_size&gt;</p>
</td>
</tr>
<tr id="row2033244801010"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p4332164821010"><a name="p4332164821010"></a><a name="p4332164821010"></a>Parameters</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p472394917248"><a name="p472394917248"></a><a name="p472394917248"></a>&lt;dim&gt;: Feature vector dimension. The default value is "256".</p>
<p id="p1572494913240"><a name="p1572494913240"></a><a name="p1572494913240"></a>&lt;nlist1&gt;: Number of first-level cluster centroids. The default value is "1024".</p>
<p id="p1972464912414"><a name="p1972464912414"></a><a name="p1972464912414"></a>&lt;nprobe1&gt;: Number of first-level candidate buckets dispatched for each retrieval calculation. The default value is "[72]".</p>
<p id="p75549202383"><a name="p75549202383"></a><a name="p75549202383"></a>&lt;nprobe2&gt;: Number of second-level candidate buckets dispatched for each retrieval calculation. The default value is "[64, 296]".</p>
<p id="p193458505381"><a name="p193458505381"></a><a name="p193458505381"></a>&lt;sub_dim1&gt;: Dimensionality after first-level reduction during retrieval. The default value is "32".</p>
<p id="p686818358392"><a name="p686818358392"></a><a name="p686818358392"></a>&lt;segment&gt;: Number of data segments searched from nprobe2 during retrieval. The default value is "[512, 1000, 1504]".</p>
<p id="p14215717122817"><a name="p14215717122817"></a><a name="p14215717122817"></a>&lt;pool_size&gt;: Process pool size for multi-process scheduling during batch operator generation. The default value is "16".</p>
<p id="p2216353369"><a name="p2216353369"></a><a name="p2216353369"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row1333284817107"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p103329483101"><a name="p103329483101"></a><a name="p103329483101"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p3370256123818"><a name="p3370256123818"></a><a name="p3370256123818"></a>Running this command generates a set of AI Core and AICPU operator model files for VSTAR retrieval. You need to modify the parameters in the command yourself. For example, to generate operators for <term>Atlas inference products</term> with a dimension of 256, 1024 cluster centroids, 72 first-level candidate buckets, 64 second-level candidate buckets, a reduced dimension of 32, and 512 search segments: python3 vstar_generate_models.py --dim 256 --nlistL1 1024 --subDimL1 32 --nProbeL1 72 --nProbeL2 64 --segmentNumL3 512</p>
</td>
</tr>
<tr id="row1827720142259"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p327810141252"><a name="p327810141252"></a><a name="p327810141252"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul144021537172510"></a><a name="ul144021537172510"></a><ul id="ul144021537172510"><li><code>dim</code> ∈ {128, 256, 512, 1024}.</li><li><code>nlist1</code> ∈ {256, 512, 1024}.</li><li><code>sub_dim1</code> ∈ {32, 64, 128}. <code>sub_dim1</code> must be less than <code>dim</code>.</li><li><code>nprobe1</code> ∈ (16, nlist1]. <code>nprobe1</code> is a list of int values, and each value in the list must be a multiple of 8.</li><li><code>nprobe2</code> ∈ (16, nprobe1 * n]. When <code>dim</code> is 1024, <code>n</code> is 16. For other dimensions, <code>n</code> is 32. <code>nprobe2</code> is a list of int values, and each value in the list must be a multiple of 8.</li><li><code>segment</code> ∈ (100, 5000]. <code>segment</code> is a list of int values, and each value must be a multiple of 8.</li><li><code>pool_size</code> ∈ [1, 32]. Before running the script, determine the maximum number of processes supported by the host machine and set it appropriately.</li></ul>
</td>
</tr>
</tbody>
</table>

**Involved Algorithms<a name="section16686174317488"></a>**

[AscendIndexVStar](./api/02_approximate_retrieval/11_AscendIndexVStar.md#ascendindexvstar)

[AscendIndexGreat](./api/02_approximate_retrieval/12_AscendIndexGreat.md#ascendindexgreat)

#### IVFFLAT<a name="en-us_TOPIC_0000002478096638"></a>

<a name="table4331184817108"></a>
<table><tbody><tr id="row1433117485104"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p733211482108"><a name="p733211482108"></a><a name="p733211482108"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p828815402278"><a name="p828815402278"></a><a name="p828815402278"></a>python3 ivfflat_generate_model.py -d &lt;dim&gt; -c &lt;coarse_centroid_num&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row2033244801010"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p4332164821010"><a name="p4332164821010"></a><a name="p4332164821010"></a>Parameters</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p472394917248"><a name="p472394917248"></a><a name="p472394917248"></a>&lt;dim&gt;: Feature vector dimension. The default value is "128".</p>
<p id="p1572494913240"><a name="p1572494913240"></a><a name="p1572494913240"></a>&lt;coarse_centroid_num&gt;: Number of first-level cluster centroids. The default value is "<span id="ph1658923911236"><a name="ph1658923911236"></a><a name="ph1658923911236"></a>1024</span>".</p>
<p id="p1149272010268"><a name="p1149272010268"></a><a name="p1149272010268"></a>&lt;core_num&gt;: Number of <span id="ph129021410310"><a name="ph129021410310"></a><a name="ph129021410310"></a>AI Cores on the Ascend AI Processor</span>. The default value is "40". If this parameter is not specified, it is configured according to &lt;npu_type&gt;: when &lt;npu_type&gt; is 910B3, &lt;core_num&gt; is 40.</p>
<p id="p1849218206267"><a name="p1849218206267"></a><a name="p1849218206267"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is "0". No additional configuration is required.</p>
<p id="p849220202265"><a name="p849220202265"></a><a name="p849220202265"></a>&lt;pool_size&gt;: Process pool size for multi-process scheduling during batch operator generation. The default value is "10".</p>
<p id="p174921720102611"><a name="p174921720102611"></a><a name="p174921720102611"></a>&lt;npu_type&gt;: Hardware form factor. Currently, &lt;npu_type&gt; supports <span id="ph996833614580"><a name="ph996833614580"></a><a name="ph996833614580"></a><term>Atlas A2 Inference products</term>, <term>Atlas A3 Inference products</term>, and <term>Ascend 950 products</term></span>. The default value is "910B4". If you cannot determine the specific npu_type, run the <strong id="b1611533911102"><a name="b1611533911102"></a><a name="b1611533911102"></a>npu-smi info</strong> command on the server where the <span id="ph16510123015103"><a name="ph16510123015103"></a><a name="ph16510123015103"></a>Ascend AI Processor</span> is installed. Remove the last digit from the reported "Name"; the remaining value is the value of npu_type. For <span id="ph1411710191414"><a name="ph1411710191414"></a><a name="ph1411710191414"></a>Atlas 800I A3 Supernode Servers</span>, run the <strong id="b12401152416117"><a name="b12401152416117"></a><a name="b12401152416117"></a>npu-smi info -t board -i 0 -c 0</strong> command to obtain <strong>NPU Name</strong>. <strong>910_</strong><strong>NPU Name</strong> is the value of npu_type. For Ascend 950 Supernode Servers, set npu_type to "Ascend950PR".</p>
<p id="p2216353369"><a name="p2216353369"></a><a name="p2216353369"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row1333284817107"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p103329483101"><a name="p103329483101"></a><a name="p103329483101"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p3370256123818"><a name="p3370256123818"></a><a name="p3370256123818"></a>Running this command generates a set of operator model files. You need to modify the parameters in the command yourself. For example, to generate operators for Atlas 800I A2 with a dimension of 256 and 1024 cluster centroids: python3 ivfflat_generate_model.py -c 1024 -t 910B4</p>
</td>
</tr>
<tr id="row1827720142259"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p327810141252"><a name="p327810141252"></a><a name="p327810141252"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul144021537172510"></a><a name="ul144021537172510"></a><ul id="ul144021537172510"><li><code>dim</code> ∈ {64, 128, 256, 384, 512}.</li><li><code>&lt;coarse_centroid_num&gt;</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}</li><li><code>0 ≤ &lt;pool_size&gt; ≤ 32</code></li></ul>
</td>
</tr>
</tbody>
</table>

**Involved Algorithms<a name="section16686174317488"></a>**

[AscendIndexIVFFlat](./api/02_approximate_retrieval/14_AscendIndexIVFFlat.md#ascendindexivfflat)

#### IVFPQ<a name="en-us_TOPIC_0000002478096638"></a>

<a name="table4331184817108"></a>
<table><tbody><tr id="row1433117485104"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p733211482108"><a name="p733211482108"></a><a name="p733211482108"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p828815402278"><a name="p828815402278"></a><a name="p828815402278"></a>python3 ivfpq_generate_model.py -d &lt;dim&gt; -c &lt;nlist&gt; --cores &lt;core_num&gt; -m &lt;m&gt; -n &lt;nbit&gt; -topK &lt;topK&gt; -b &lt;blockNum&gt; -p &lt;process_id&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row2033244801010"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p4332164821010"><a name="p4332164821010"></a><a name="p4332164821010"></a>Parameters</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p472394917248"><a name="p472394917248"></a><a name="p472394917248"></a>&lt;dim&gt;: Feature vector dimension. The default value is "128".</p>
<p id="p1572494913240"><a name="p1572494913240"></a><a name="p1572494913240"></a>&lt;nlist&gt;: Number of first-level cluster centroids. The default value is "<span id="ph1658923911236"><a name="ph1658923911236"></a><a name="ph1658923911236"></a>1024</span>".</p>
<p id="p1149272010268"><a name="p1149272010268"></a><a name="p1149272010268"></a>&lt;core_num&gt;: Number of <span id="ph129021410310"><a name="ph129021410310"></a><a name="ph129021410310"></a>AI Cores on the Ascend AI Processor</span>. The default value is "40". If this parameter is not specified, it is configured according to &lt;npu_type&gt;.</p>
<p id="p849220202265"><a name="p849220202265"></a><a name="p849220202265"></a>&lt;m&gt;: Number of subspaces. The default value is "4".</p>
<p id="p849220202265"><a name="p849220202265"></a><a name="p849220202265"></a>&lt;nbit&gt;: Number of bits in the quantization centroid for each subspace. The default value is "8", and no additional configuration is required. It also determines the number of codebook centroids, <code>ksub = 1 &lt;&lt; nbit</code>. When <code>nbit</code> is 8, <code>ksub</code> is 256</p>
<p id="p849220202265"><a name="p849220202265"></a><a name="p849220202265"></a>&lt;topK&gt;: Number of nearest candidate vectors returned for each query vector. The default value is "320", and no additional configuration is required.</p>
<p id="p849220202265"><a name="p849220202265"></a><a name="p849220202265"></a>&lt;blockNum&gt;: Number of candidate vector blocks to process. The default value is "128", and no additional configuration is required.</p>
<p id="p1849218206267"><a name="p1849218206267"></a><a name="p1849218206267"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is "0". No additional configuration is required.</p>
<p id="p174921720102611"><a name="p174921720102611"></a><a name="p174921720102611"></a>&lt;npu_type&gt;: Hardware form factor. The current default value is "Ascend950PR". If you cannot determine the specific npu_type, run the <strong id="b1611533911102"><a name="b1611533911102"></a><a name="b1611533911102"></a>npu-smi info</strong> command on the server where the <span id="ph16510123015103"><a name="ph16510123015103"></a><a name="ph16510123015103"></a>Ascend AI Processor</span> is installed. Remove the last digit from the reported "Name"; the remaining value is the value of npu_type. For <span id="ph1411710191414"><a name="ph1411710191414"></a><a name="ph1411710191414"></a>Atlas 800I A3 Supernode Servers</span>, run the <strong id="b12401152416117"><a name="b12401152416117"></a><a name="b12401152416117"></a>npu-smi info -t board -i 0 -c 0</strong> command to obtain <strong>NPU Name</strong>. <strong>910_</strong><strong>NPU Name</strong> is the value of npu_type.</p>
<p id="p2216353369"><a name="p2216353369"></a><a name="p2216353369"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row1333284817107"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p103329483101"><a name="p103329483101"></a><a name="p103329483101"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p3370256123818"><a name="p3370256123818"></a><a name="p3370256123818"></a>Running this command generates a set of operator model files. You need to modify the parameters in the command yourself. For example, to generate operators for Atlas A5 with a dimension of 128, 1024 cluster centroids, 4 subspaces, and 8 bits: python3 ivfpq_generate_model.py -d 128 -c 1024 -m 4 -n 8 -t Ascend950PR</p>
</td>
</tr>
<tr id="row1827720142259"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p327810141252"><a name="p327810141252"></a><a name="p327810141252"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul144021537172510"></a><a name="ul144021537172510"></a><ul id="ul144021537172510"><li><code>dim</code> ∈ {128}</li><li><code>nlist</code> ∈ {1024, 2048, 4096, 8192, 16384}</li><li><code>m</code> ∈ {2, 4, 8, 16}</li><li><code>n</code> ∈ {8}</li></ul>
</td>
</tr>
</tbody>
</table>

**Involved Algorithms<a name="section16686174317488"></a>**

[AscendIndexIVFPQ](./api/02_approximate_retrieval/15_AscendIndexIVFPQ.md#ascendindexivfpq)

#### IVFRaBitQ<a name="en-us_TOPIC_0000002513317244"></a>

<a name="table1844216303913"></a>
<table><tbody><tr id="row124438353916"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p1944314323914"><a name="p1944314323914"></a><a name="p1944314323914"></a>Usage</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p033194912546"><a name="p033194912546"></a><a name="p033194912546"></a>python3 ivfrabitq_generate_model.py -d &lt;dim&gt; -c &lt;coarse_centroid_num&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt; -m &lt;metric_type&gt;</p>
</td>
</tr>
<tr id="row1344373183918"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p244314363910"><a name="p244314363910"></a><a name="p244314363910"></a>Parameters</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p472394917248"><a name="p472394917248"></a><a name="p472394917248"></a>&lt;dim&gt;: Feature vector dimension. The default value is "128".</p>
<p id="p1572494913240"><a name="p1572494913240"></a><a name="p1572494913240"></a>&lt;coarse_centroid_num&gt;: Number of first-level cluster centroids. The default value is "16384".</p>
<p id="p113021935175610"><a name="p113021935175610"></a><a name="p113021935175610"></a>&lt;core_num&gt;: Number of <span id="ph83021535135613"><a name="ph83021535135613"></a><a name="ph83021535135613"></a>AI Cores on the Ascend AI Processor</span>. The default value is "40". If this parameter is not specified, it is configured according to &lt;npu_type&gt;: when &lt;npu_type&gt; is 910B3, &lt;core_num&gt; is 40.</p>
<p id="p10302153517569"><a name="p10302153517569"></a><a name="p10302153517569"></a>&lt;process_id&gt;: Process ID for multi-process scheduling during batch operator generation. The default value is "0". No additional configuration is required.</p>
<p id="p849220202265"><a name="p849220202265"></a><a name="p849220202265"></a>&lt;pool_size&gt;: Process pool size for multi-process scheduling during batch operator generation. The default value is "10".</p>
<p id="p174921720102611"><a name="p174921720102611"></a><a name="p174921720102611"></a>&lt;npu_type&gt;: Hardware form factor. Currently, &lt;npu_type&gt; supports <term>Atlas A2 Inference products</term>, <term>Atlas A3 Inference products</term>, with a default value of "910B4". If you cannot determine the specific npu_type, run the <strong id="b1611533911102"><a name="b1611533911102"></a><a name="b1611533911102"></a>npu-smi info</strong> command on the server where the <span id="ph16510123015103"><a name="ph16510123015103"></a><a name="ph16510123015103"></a>Ascend AI Processor</span> is installed. Remove the last digit from the reported "Name"; the remaining value is the value of npu_type. For <span id="ph1411710191414"><a name="ph1411710191414"></a><a name="ph1411710191414"></a>Atlas 800I A3 Supernode Servers</span>, run the <strong id="b12401152416117"><a name="b12401152416117"></a><a name="b12401152416117"></a>npu-smi info -t board -i 0 -c 0</strong> command to obtain <strong>NPU Name</strong>. <strong>910_</strong><strong>NPU Name</strong> is the value of npu_type.</p>
<p id="p116211110135912"><a name="p116211110135912"></a><a name="p116211110135912"></a>&lt;metric_type&gt;: Vector calculation mode, used to explicitly specify whether to calculate using "L2" or "IP" distance. The default value is "L2".</p>
<p id="p4302153511569"><a name="p4302153511569"></a><a name="p4302153511569"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row44439314393"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p1444316303912"><a name="p1444316303912"></a><a name="p1444316303912"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p144431434393"><a name="p144431434393"></a><a name="p144431434393"></a>Running this command generates a set of operator model files. You need to modify the parameters in the command yourself. For example, to generate operators for Atlas 800I A2 with a dimension of 128, 1024 cluster centroids, and L2 distance: python3 ivfrabitq_generate_model.py -d 128 -c 1024 -t 910B4 -m L2</p>
</td>
</tr>
<tr id="row19923139133710"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p68850165377"><a name="p68850165377"></a><a name="p68850165377"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul16885121613379"></a><a name="ul16885121613379"></a><ul id="ul16885121613379"><li><code>dim</code> ∈ {128}</li><li><code>&lt;coarse_centroid_num&gt;</code> ∈ {1024, 2048, 4096, 8192, 10048, 16384, 32768}</li><li><code>0 ≤ &lt;pool_size&gt; ≤ 32</code></li></ul>
</td>
</tr>
</tbody>
</table>

**Involved Algorithms<a name="section16686174317488"></a>**

[AscendIndexIVFRaBitQ](./api/02_approximate_retrieval/16_AscendIndexIVFRaBitQ.md#ascendindexivfrabitq)

**Runtime Diagnostics (Development and Debugging)<a name="ivfrabitq-runtime-debug-ref"></a>**

When troubleshooting coarse centroid upload or L1 reranking issues, you can use debugging environment variables to locate the fault in stages. These variables are disabled by default and do not affect performance. For details about the environment variables, see [Appendix](./09_appendix.md#ivfrabitq-debug-env). For operation steps and log interpretation, see [Common Operations — IVFRaBitQ Runtime Diagnostics](./08_common_operations.md#ivfrabitq-runtime-debug).

#### VSTAR Codebook File Generation<a name="en-us_TOPIC_0000002008789068"></a>

##### Overall Description<a name="en-us_TOPIC_0000002045184529"></a>

**Environment Setup<a name="section12757124191817"></a>**

The environment dependencies are as follows:

- nnae (version >= 8.0.0; included in the toolkit package starting from version 8.5.0)
- python (version >= 3.9)
- torch (version >= 2.0.1)
- torch_npu (version >= 2.0.1.post4)

- numpy (version >= 1.26.4)
- scikit-learn (version >= 1.4.1.post1)
- tqdm (version >= 4.66.1)

You can install torch, TorchNPU, numpy, scikit-learn, and tqdm using the **pip install** command. The following is an example.

```bash
pip install numpy tqdm scikit-learn torch_npu torch
```

Versions earlier than CANN 8.5.0 require a separate nnae installation. Follow these steps:

1. Download the [nnae](https://www.hiascend.com/developer/download/community/result?module=cann&product=2&model=17) package.
2. Run the following command to add execute permissions.

    ```bash
    chmod u+x ./Ascend-cann-nnae_{version}_linux-{arch}.run
    ```

3. Run the following command to install it.

    ```bash
    ./Ascend-cann-nnae_{version}_linux-{arch}.run --install
    ```

4. Set the environment variables according to the installation prompts.

    ```bash
    source /{nnae_installation_path}/nnae/set_env.sh
    ```

**Notes<a name="section15462185871819"></a>**

- If you encounter the following error when importing torch with TorchNPU:

    ```text
    .../libgomp.so: cannot allocate memory in static TLS block
    ```

  Run `export LD_PRELOAD=.../libgomp.so`, where the path is the `libgomp.so` path shown in the error message.

- If pip cannot install the following dependencies when installing numpy:

    ```text
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behavior is the source of the following dependency conflicts.
    auto-tune 0.1.0 requires decorator, which is not installed.
    dataflow 0.0.1 requires jinja2, which is not installed.
    opc-tool 0.1.0 requires attrs, which is not installed.
    opc-tool 0.1.0 requires decorator, which is not installed.
    opc-tool 0.1.0 requires psutil, which is not installed.
    schedule-search 0.0.1 requires absl-py, which is not installed.
    schedule-search 0.0.1 requires decorator, which is not installed.
    te 0.4.0 requires attrs, which is not installed.
    te 0.4.0 requires cloudpickle, which is not installed.
    te 0.4.0 requires decorator, which is not installed.
    te 0.4.0 requires ml-dtypes, which is not installed.
    te 0.4.0 requires psutil, which is not installed.
    te 0.4.0 requires scipy, which is not installed.
    te 0.4.0 requires tornado, which is not installed.
    ```

  Run the following command.

    ```bash
    pip install attrs cloudpickle decorator jinja2 ml-dtypes psutil scipy tornado absl-py
    ```

- If you encounter the following issue when training the codebook:

    ```text
    OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata.
    Segmentation fault (core dumped)
    ```

  Run:

    ```bash
    export OPENBLAS_NUM_THREADS=1
    ```

    This environment variable may affect performance. After codebook training is complete, you are advised to restore it to the preset value.

- Details about the `--useOfflineCompile` option:

    Online operator compilation takes longer than offline operator compilation. The `--useOfflineCompile` option controls whether the operator package and offline operator compilation are used to improve performance. When enabled, precompiled offline operator packages are used for execution. This method requires you to install the single-operator package in advance. Follow these steps to install the operator package:

    1. Download the [operator package](https://www.hiascend.com/developer/download/community/result?module=cann&product=2&model=17).
    2. Run the following command to add execute permissions.
        - Versions earlier than CANN 8.5.0

            ```bash
            chmod u+x ./Ascend-cann-kernels-{chip_type}_{version}_linux-{arch}.run
            ```

        - CANN 8.5.0 and later

            ```bash
            chmod u+x ./Ascend-cann-{chip_type}-ops_{version}_linux-{arch}.run
            ```

    3. Run the following command to install the operator package.
        - Versions earlier than CANN 8.5.0

            ```bash
            ./Ascend-cann-kernels-{chip_type}_{version}_linux-{arch}.run --install
            ```

        - CANN 8.5.0 and later

            ```bash
            ./Ascend-cann-{chip_type}-ops_{version}_linux-{arch}.run --install
            ```

    4. Set the environment variables according to the installation prompts.
        - Versions earlier than CANN 8.5.0

            ```bash
            source /{kernels_installation_path}/kernels/set_env.sh
            ```

        - CANN 8.5.0 and later

            ```bash
            source /usr/local/Ascend/cann/set_env.sh
            ```

##### Codebook Training Script<a name="en-us_TOPIC_0000002008865568"></a>

Training involves the `vstar_train_codebook.py` script. The training script is located in the `tools/train` folder under the installation directory. Note that the Python version is 3.9.

<a name="table48723587152"></a>
<table><tbody><tr id="row4899125881510"><th class="firstcol" valign="top" width="13.62%" id="mcps1.1.3.1.1"><p id="p1089905812153"><a name="p1089905812153"></a><a name="p1089905812153"></a>Command Reference</p>
</th>
<td class="cellrowborder" valign="top" width="86.38%" headers="mcps1.1.3.1.1 "><p id="p989945810152"><a name="p989945810152"></a><a name="p989945810152"></a>python3 vstar_train_codebook.py --dataPath &lt;data_path&gt; --dim &lt;dim&gt; --codebookPath &lt;codebook_output_dir&gt; --nlistL1 &lt;nlist1&gt; --subDimL1 &lt;sub_dim1&gt;  --device &lt;device&gt; --batchSize &lt;batch_size&gt; --sample &lt;sample&gt; --useOfflineCompile</p>
</td>
</tr>
<tr id="row13899195817158"><th class="firstcol" valign="top" width="13.62%" id="mcps1.1.3.2.1"><p id="p198995588156"><a name="p198995588156"></a><a name="p198995588156"></a>Parameter Name</p>
</th>
<td class="cellrowborder" valign="top" width="86.38%" headers="mcps1.1.3.2.1 "><p id="p112782011172018"><a name="p112782011172018"></a><a name="p112782011172018"></a>&lt;data_path&gt;: Path to the raw data used for codebook training. The data must exist. This parameter is required.</p>
<p id="p4899195861514"><a name="p4899195861514"></a><a name="p4899195861514"></a>&lt;dim&gt;: Feature vector dimension. Keep it consistent with the &lt;dim&gt; used to generate the VSTAR training operator model file. The default value is <span class="parmvalue" id="parmvalue9207155164317"><a name="parmvalue9207155164317"></a><a name="parmvalue9207155164317"></a>"256"</span>.</p>
<p id="p9584201083120"><a name="p9584201083120"></a><a name="p9584201083120"></a>&lt;codebook_output_dir&gt;: Path for storing the generated codebook file. The generated codebook file is output to this directory. You must ensure that the directory exists and that the user running the program has write permission for this directory. For security hardening reasons, the directory hierarchy must not contain symbolic links.</p>
<p id="p689915891520"><a name="p689915891520"></a><a name="p689915891520"></a>&lt;nlist1&gt;: Number of first-level cluster centroids. Keep it consistent with the &lt;nlist1&gt; used to generate the VSTAR training operator model file. The default value is "1024".</p>
<p id="p193458505381"><a name="p193458505381"></a><a name="p193458505381"></a>&lt;sub_dim1&gt;: Dimensionality after first-level reduction during retrieval. Keep it consistent with the &lt;sub_dim1&gt; used to generate the VSTAR training operator model file. The default value is "32".</p>
<p id="p68991758111519"><a name="p68991758111519"></a><a name="p68991758111519"></a>&lt;device&gt;: Device logical ID. Training is performed on the specified Device. The default value is "1".</p>
<p id="p089915814155"><a name="p089915814155"></a><a name="p089915814155"></a>&lt;batch_size&gt;: Batch size used for training. The parameter range is (0, 10240]. The default value is <span class="parmvalue" id="parmvalue17103111818276"><a name="parmvalue17103111818276"></a><a name="parmvalue17103111818276"></a>"10240"</span>.</p>
<p id="p168991358141520"><a name="p168991358141520"></a><a name="p168991358141520"></a>&lt;sample&gt;: Sampling rate of the raw samples used for training. 0 &lt; ratio ≤ 1.0. The default value is <span class="parmvalue" id="parmvalue497014364412"><a name="parmvalue497014364412"></a><a name="parmvalue497014364412"></a>"1.0"</span>.</p>
<p id="p1581627174815"><a name="p1581627174815"></a><a name="p1581627174815"></a>--useOfflineCompile: Controls whether to use the operator package dependency and offline operator compilation to improve performance. Disabled by default. To enable it, add this option to the end of the command line. For details, see VSTAR Codebook File Generation - Overall Description - Details about the --useOfflineCompile option.</p>
<p id="p575632015367"><a name="p575632015367"></a><a name="p575632015367"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row789917582151"><th class="firstcol" valign="top" width="13.62%" id="mcps1.1.3.3.1"><p id="p789955811157"><a name="p789955811157"></a><a name="p789955811157"></a>Usage Instructions</p>
</th>
<td class="cellrowborder" valign="top" width="86.38%" headers="mcps1.1.3.3.1 "><a name="ul18156187102113"></a><a name="ul18156187102113"></a><ul id="ul18156187102113"><li>&lt;data_path&gt;: The size of the raw data must be no greater than 10 million 1024-dimensional vectors, that is, 10,000,000 * 1024 * 4 = 40,960,000,000.</li><li>Running this command generates a new directory, <code>codebook_&lt;dim&gt;_&lt;nlist1&gt;_&lt;sub_dim1&gt;.bin</code>, under the directory specified by &lt;codebook_output_dir&gt;. This is the codebook file required by AscendIndexVStar and AscendIndexGreat.</li><li>If the codebook file already exists, it is overwritten. In this case, the user running the program must be the file owner.</li><li>Before running the training to generate the codebook, first refer to VSTAR to generate the training operator model file.</li></ul>
</td>
</tr>
</tbody>
</table>

#### (Optional) Generate Codebook Files in Python<a name="en-us_TOPIC_0000001649848464"></a>

##### IVFSP Training Script<a name="en-us_TOPIC_0000001585736180"></a>

**Environment Configuration**

The environment dependencies are:

- numpy (version > 1.16.0)
- tqdm (version ≥ 4.65.0)
- faiss-cpu (version = 1.10.0)

Run the following command to install the dependencies.

```bash
pip install numpy tqdm faiss-cpu==1.10.0
```

Run the following command to set the environment variables.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**Training Script Execution**

Index SDK provides two methods for training:

- Use the [trainCodeBook](./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#traincodebook) API of the IVFSP algorithm for training. This method is recommended.
- Use the `ivfsp_train_codebook.py` script for training. The Python version is 3.9.11. The sample script `ivfsp_train_codebook_example.sh` is provided in the `tools/train` directory. You can modify the parameter values in the script and run it to generate the codebook.

<a name="table48723587152"></a>
<table><tbody><tr id="row4899125881510"><th class="firstcol" valign="top" width="13.63%" id="mcps1.1.3.1.1"><p id="p1089905812153"><a name="p1089905812153"></a><a name="p1089905812153"></a>Command Reference</p>
</th>
<td class="cellrowborder" valign="top" width="86.37%" headers="mcps1.1.3.1.1 "><p id="p989945810152"><a name="p989945810152"></a><a name="p989945810152"></a>python3 ivfsp_train_codebook.py --dim &lt;dim&gt; --nonzero_num &lt;nonzero_num&gt; --nlist &lt;nlist&gt; --num_iter &lt;num_iter&gt; --device &lt;device&gt; --batch_size &lt;batch_size&gt; --code_num &lt;code_num&gt; --ratio &lt;ratio&gt; --learn_data_path &lt;learn_data_path&gt; --codebook_output_dir &lt;codebook_output_dir&gt; --train_model_dir &lt;train_model_dir&gt;</p>
</td>
</tr>
<tr id="row13899195817158"><th class="firstcol" valign="top" width="13.63%" id="mcps1.1.3.2.1"><p id="p198995588156"><a name="p198995588156"></a><a name="p198995588156"></a>Parameter Name</p>
</th>
<td class="cellrowborder" valign="top" width="86.37%" headers="mcps1.1.3.2.1 "><p id="p4899195861514"><a name="p4899195861514"></a><a name="p4899195861514"></a>&lt;dim&gt;: Feature vector dimension. Keep it consistent with the &lt;dim&gt; used to generate the IVFSP training operator model file. It must be greater than 0.</p>
<p id="p1226910612333"><a name="p1226910612333"></a><a name="p1226910612333"></a>&lt;nonzero_num&gt;: Number of non-zero dimensions after feature vector compression. Keep it consistent with the &lt;low_dim&gt; used to generate the IVFSP training operator model file. It must be greater than 0.</p>
<p id="p689915891520"><a name="p689915891520"></a><a name="p689915891520"></a>&lt;nlist&gt;: Number of cluster centroids. Keep it consistent with the &lt;k&gt; used to generate the IVFSP training operator model file. It must be greater than 0.</p>
<p id="p089935861513"><a name="p089935861513"></a><a name="p089935861513"></a>&lt;num_iter&gt;: Number of training iterations. The default value is 20. Setting it too large increases the training time. It must be greater than 0.</p>
<p id="p68991758111519"><a name="p68991758111519"></a><a name="p68991758111519"></a>&lt;device&gt;: Device logical ID. Training is performed on the specified Device. The default value is "0".</p>
<p id="p089915814155"><a name="p089915814155"></a><a name="p089915814155"></a>&lt;batch_size&gt;: Batch size used for training. Keep it consistent with the &lt;batch_size&gt; used to generate the IVFSP training operator model file. It must be greater than 0 and less than or equal to 32768. The default value is <span class="parmvalue" id="parmvalue17103111818276"><a name="parmvalue17103111818276"></a><a name="parmvalue17103111818276"></a>"32768"</span>.</p>
<p id="p9899195814150"><a name="p9899195814150"></a><a name="p9899195814150"></a>&lt;code_num&gt;: Maximum number of samples processed against the codebook at a time. It must be a power of 2. Keep it consistent with the &lt;codebook_batch_size&gt; used to generate the IVFSP training operator model file. It must be greater than 0 and less than or equal to 32768. The default value is <span class="parmvalue" id="parmvalue13486113643711"><a name="parmvalue13486113643711"></a><a name="parmvalue13486113643711"></a>"32768"</span>.</p>
<p id="p168991358141520"><a name="p168991358141520"></a><a name="p168991358141520"></a>&lt;ratio&gt;: Sampling rate of the raw samples used for training. 0 &lt; ratio ≤ 1.0. The default value is 1.0.</p>
<p id="p1889985811157"><a name="p1889985811157"></a><a name="p1889985811157"></a>&lt;learn_data_path&gt;: Path to the raw feature file used for training. The <code>bin</code> and <code>npy</code> formats are supported. For the <code>bin</code> format, data is stored in row-major order and uses the <code>float32</code> data type.</p>
<p id="p1089935810157"><a name="p1089935810157"></a><a name="p1089935810157"></a>&lt;codebook_output_dir&gt;: Directory where the generated codebook file is output. You must ensure that the directory exists and that the user running the program has write permission for the directory. For security hardening reasons, the directory hierarchy must not contain symbolic links.</p>
<p id="p168997585150"><a name="p168997585150"></a><a name="p168997585150"></a>&lt;train_model_dir&gt;: Directory containing the IVFSP training operator model file.</p>
<p id="p11852192143213"><a name="p11852192143213"></a><a name="p11852192143213"></a>--help | -h: Displays help information.</p>
</td>
</tr>
<tr id="row789917582151"><th class="firstcol" valign="top" width="13.63%" id="mcps1.1.3.3.1"><p id="p789955811157"><a name="p789955811157"></a><a name="p789955811157"></a>Usage Instructions</p>
</th>
<td class="cellrowborder" valign="top" width="86.37%" headers="mcps1.1.3.3.1 "><a name="ul18156187102113"></a><a name="ul18156187102113"></a><ul id="ul18156187102113"><li>Running this command generates the <code>codebook_&lt;dim&gt;_&lt;nonzero_num&gt;_&lt;nlist&gt;.bin</code> and <code>codebook_&lt;dim&gt;_&lt;nonzero_num&gt;_&lt;nlist&gt;.npy</code> files in the directory specified by <code>&lt;codebook_output_dir&gt;</code>. The <code>codebook_&lt;dim&gt;_&lt;nonzero_num&gt;_&lt;nlist&gt;.bin</code> file is the codebook file required by AscendIndexIVFSP.</li><li>If the codebook file already exists, it is overwritten. In this case, the user running the program must be the file owner.</li><li>Before running the training to generate the codebook, first refer to the IVFSP training operator model file generation instructions to generate the training operator model file.</li><li>The size of the data specified by <code>learn_data_path</code> must be greater than or equal to <code>nonzero_num * nlist * sizeof(float32)</code> bytes.</li></ul>
</td>
</tr>
</tbody>
</table>

##### Dimensionality Reduction Training Script<a name="en-us_TOPIC_0000001681635905"></a>

**Environment Dependencies<a name="section162431329141010"></a>**

- Install Python 3.9. Python 3.9, Python 3.10, and Python 3.11 are supported. Python 3.9 is recommended.
- Install Faiss 1.10.0. You can install it using the **pip install** command. The following is an example.

    ```bash
    pip install faiss-cpu==1.10.0
    ```

- Install torch_cpu and TorchNPU. For details, see the [link](https://gitcode.com/Ascend/pytorch). Select and install the corresponding versions according to the version compatibility table.

**Training the Model<a name="section8422152014206"></a>**

The default path for the scripts involved in this section is `tools/train/reduction`.

1. **Train the model.**

    ```bash
    python3 call_train.py --dataset_dir=Dataset_Dir --val_dataset_dir=./valid --generate_val=True --save_path=./modelsDr --dim=512 --npu=0 --ratio=4 --metric=L2 --mode=train --train_size=100000 --epochs=20 --train_batch_size=8192 --infer_batch_size=128 --learning_rate=0.0005 --log_stride=500 --construct_neighbors=100 --queries_validation=1000
    ```

    | Parameter | Description |
    | --------- | ----------- |
    | dataset_dir | Dataset path. The type is string. This parameter is required. The current implementation reads `base.npy`, `query.npy`, and `gt.npy` by default. If the dataset uses other names, you can implement dataset reading yourself and modify the corresponding line where `get_train_data` is located in the script. For example. The original code is:<br>`# load dataset demo before training, modify here if you want to load your own dataset        #####################################################################        learn, base = get_train_data(args.dataset_dir, args.train_size)        #####################################################################`<br>It can be modified to:<br>`# load dataset demo before training, modify here if you want to load your own dataset        #####################################################################        # learn, base = get_train_data(args.dataset_dir, args.train_size)        learn = np.fromfile(YOUR_LEARN_DATASET_DIR, dtype=np.float32).reshape((-1, YOUR_DATA_DIM))        base = np.fromfile(YOUR_BASE_DATASET_DIR, dtype=np.float32).reshape((-1, YOUR_DATA_DIM))        #####################################################################` |
    | val_dataset_dir | Valid when `generate_val` is set to True. Path for storing the generated validation set. The type is string. The default value is `./validation/`. |
    | generate_val | Whether to generate a validation set. Set this parameter to True for the first training. The type is bool. The default value is False. |
    | save_path | Path for storing the model. The type is string. This parameter is required. |
    | dim | Optional. Dataset dimension. The value range is [96, 128, 200, 256, 512, 2048]. The type is int. The default value is 512. |
    | npu | Device ID used for training. The type is int. Only single-card training is supported. CPU training is used by default. |
    | ratio | Optional. Dimensionality reduction ratio. The value range is [2, 4, 8, 16]. The type is int. The default value is 8. |
    | metric | Distance metric used for model training. L2 or IP can be selected. The type is string. The default value is L2. |
    | mode | Optional. The value range is ["train", "infer", "test"], but currently only "train" is supported. The default value is "train". No modification is required. |
    | train_size | Training set size. The value must be less than the number of samples in the entire dataset. It is used to randomly sample part of the dataset for training when reading the dataset. The type is int. If you implement dataset reading yourself, sample data according to `train_size` to prevent the training speed from becoming too slow. The default value is 100000. When modifying this value, ensure that it is greater than 0. |
    | epochs | Number of training epochs. The type is int. Setting the number of iterations too large significantly increases the training time. The default value is 30. When modifying this value, ensure that it is greater than 0. |
    | train_batch_size | Batch size used for training. The default value is "8192". The type is int. When modifying this value, ensure that it is greater than 0. |
    | infer_batch_size | Batch size used for inference. The default value is "128". The type is int. When modifying this value, ensure that it is greater than 0. |
    | learning_rate | Learning rate. The default value is "0.0005". The type is float. When modifying this value, ensure that it is greater than 0. |
    | log_stride | Training log printing interval (step). The default value is "500". The type is int. When modifying this value, ensure that it is greater than 0. |
    | construct_neighbors | Range of nearest neighbors selected when constructing the training set. It is used to construct the special training set structure required for dimensionality reduction. The default value is "100". Modify it according to the number of faces corresponding to each person in the dataset. The type is int. When modifying this value, ensure that it is greater than 0. |
    | queries_validation | Number of query vectors required when constructing the validation set. The type is int. The default value is "1000". When modifying this value, ensure that it is greater than 0. |
    | --help \| -h | Displays help information. |

2. **Generate the OM Model.**

    Before running the script, run the following commands:

    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
    ```

    1. **Generate an OM model with 32-bit precision.**

        ```bash
        bash atc.sh {save_path} {om_name} {input_shape}
        ```

    2. **Generate an OM model with 16-bit precision.**

        ```bash
        bash atc_16.sh {save_path} {om_name} {input_shape}
        ```

    - `{save_path}`: Required. Path for storing the model. The file name in the path must end with `.onnx` or `.pb`. Otherwise, the script cannot obtain values such as the `"framework"` and `"input_format"` environment variables, which causes the script to fail.
    - `{om_name}`: Optional. Name of the generated OM model. The default value is the same as the ONNX model name.
    - `{input_shape}`: Optional. The default value is the input dimension of the ONNX model. The format is `actual_input_1:infer_batch_size,dim`. The default value is recommended, and modification is not recommended.
    - **`bash atc.sh`** and **`bash atc_16.sh`** support only <term>Atlas inference products</term>.1
