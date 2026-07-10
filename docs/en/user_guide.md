# Usage Guide

## Generating Operators

After you install Index SDK, follow the instructions in this section to set the operator-related environment variables and generate the operators required by the algorithms.

> [!NOTE]
>
> - The AscendIndexFlat algorithm supports online operator conversion for L2 and IP distances. If the environment variable `MX_INDEX_USE_ONLINEOP` is set to `1` (set it with `export MX_INDEX_USE_ONLINEOP=1`), the operator is converted and called online. You do not need to generate offline operators as described in this section. To use an online operator, the application must explicitly call `(void)aclFinalize()` at the end. You also need to include the header file `acl/acl.h`.
> - For algorithms that do not support online operators, setting `MX_INDEX_USE_ONLINEOP=1` causes the program to fail.

**Procedure**

1. Enter the installation directory `mxIndex-{version}`. The directory and file names are shown in [Table 1 Index SDK directories and files](#table81133951612).

    ```bash
    cd mxIndex-{version}
    ```

    **Table 1 Index SDK directories and files<a id="table81133951612"></a>**

    | Directory or File | Description |
    | -- | -- |
    | device | Contains the dynamic libraries and header files for the IndexIL algorithm. |
    | filelist.txt | Package file list. |
    | host | Search dynamic library. When you perform feature search, link to the dynamic libraries in this folder. |
    | include | API header files. |
    | lib | Search dynamic library, linked to `host/lib`. |
    | modelpath | Directory for operator `.om` files. After the operators are compiled, place the `.om` files in this folder. |
    | ops | Contains the `custom_opp_<arch>.run` script for installing search algorithm operators. |
    | script | Contains the uninstall script `uninstall.sh` for uninstalling the Index SDK package. |
    | tools | Contains the Python scripts for operator generation. |
    | version.info | Contains version-related information. |

2. Enter the `ops` directory. Before you compile operators, set the `ASCEND_HOME`, `ASCEND_VERSION`, and `ASCEND_OPP_PATH` environment variables. The default values are `~/Ascend`, `~/ascend-toolkit/latest`, and `~/Ascend/ascend-toolkit/latest/opp`, respectively.

    ```bash
    export ASCEND_HOME=~/Ascend
    export ASCEND_VERSION=~/Ascend/ascend-toolkit/latest
    export ASCEND_OPP_PATH=~/Ascend/ascend-toolkit/latest/opp
    ```

    - `ASCEND_HOME` indicates the file storage path after the CANN toolkit is installed.
    - `ASCEND_VERSION` indicates the Ascend version currently in use. If the ATC tool installation path is `/usr/local/Ascend/ascend-toolkit/latest`, you do not need to set `ASCEND_HOME` and `ASCEND_VERSION`.
    - `ASCEND_OPP_PATH` indicates the root directory of the operator library. You need write permission for this directory.

    > [!NOTE]
    >
    > `MAX_COMPILE_CORE_NUMBER` specifies the number of CPU cores available during graph compilation and is used at operator runtime. The current default value is `1`, so you do not need to set it.

3. Run the corresponding script according to the actual system architecture.

    - Arm architecture:

        ```bash
        ./custom_opp_aarch64.run
        ```

    - x86_64 architecture:

        ```bash
        ./custom_opp_x86_64.run
        ```

    You can also pass optional parameters when you run the script, as shown in [Table 2 custom_opp_{arch}.run parameter description](#table38211859291).

    **Table 2 custom_opp_{arch}.run parameter description<a id="table38211859291"></a>**

    | Parameters | Description |
    | -- | -- |
    | --help \| -h | Query help information. |
    | --info | Query package build information. |
    | --list | Query the file list. |
    | --check | Query package integrity. |
    | --quiet \|-q | Optional parameter that enables silent installation. It reduces interactive output. |
    | --nox11 | Deprecated interface with no practical effect. |
    | --noexec | Extract the package to the current directory without running the installation script. Use it with `--extract=<path>`, in the format `--noexec --extract=<path>`. |
    | --extract=\<path> | Extract the files in the package to the specified directory. You can use it with `--noexec`. |
    | --tar arg1 [arg2 ...] | Run the tar command on the package and use the parameters after tar as command arguments. For example, `--tar xvf` extracts the contents of the run installer package to the current directory. |

    > [!NOTE]
    >
    > The following parameters do not appear in the `--help` output. Do not use them directly.
    >
    > - `--xwin`: Run in xwin mode.
    > - `--phase2`: Require execution of the second step.

4. Enter the `tools` directory to generate the required operators. Before you generate operators, ensure that the relevant CANN dependencies are installed.
    - To generate only the operators required by the algorithm you use, first refer to the [Algorithm Introduction](#algorithm-introduction) section to confirm which operators need to be generated, and then refer to the [Custom Operator Introduction](#generating-operators) section to generate the corresponding operators.
    - To generate operators for all algorithms in batch, use the method shown in [Table 3 Batch generation of operators](#table03891576018).

        **Table 3 Batch generation of operators<a name="table03891576018"></a>**

        | Usage | `python3 run_generate_model.py -m <mode> -t <npu_type> -p <pipeline> -pool <pool_size>` |
        | -- | -- |
        | Parameters | <ul><li>`<mode>`: Algorithm mode. `<mode>` supports `ALL` and one or more of `Flat`, `SQ8`, `IVFSQ8`, and `INT8`. Separate multiple values with commas, for example: `python3 run_generate_model.py -m Flat,IVFSQ8`. All algorithms are selected by default, so you can run `python3 run_generate_model.py` directly.</li><li>`<npu_type>`: The chip name. - For Atlas 200/300/500 Inference Products and Atlas Inference Series products, run `npu-smi info` on the server where the Ascend AI Processor is installed. Remove the last digit from the reported `Name`, and the remaining value is `npu_type`.</li><li>For Atlas 800I A2 Inference Servers, run `npu-smi info` on the server where the Ascend AI Processor is installed. The reported `Name` is `npu_type`.</li><li>For Atlas 800I A3 Supernode Servers, run `npu-smi info -t board -i 0 -c 0` to obtain `NPU Name`. `910_` plus the `NPU Name` value is `npu_type`.</li><li>`<pipeline>`: Whether to use multi-threaded parallel pipelines to generate operator models. The default value is `true`. When set to `true`, the default `pool_size` is `32`.</li><li>`<pool_size>`: The process pool size for multi-process scheduling during batch operator generation.</li><li>`--help \| -h`: Query help information.</li></ul> |
        | Description | <ul><li>Running this command generates multiple sets of operator model files.</li><li>Before you run it, update the `para_table.xml` file in the current directory and fill in the required parameters in the table.</li><li>`1 ≤ pool_size ≤ 32`.</li></ul> |

        > [!NOTE]
        >
        > The constraint descriptions in the operator generation table represent parameter combinations that commonly appear in business scenarios. If you see exceptions when you run the tool with other parameters, refer to the [CANN ATC Offline Model Compilation Tool User Guide](https://www.hiascend.com/document/detail/zh/canncommercial/900/devaids/atctool/atlasatc_16_0001.html).

5. Prepare the operator model files.

    - You can configure the operator model file directory through the `MX_INDEX_MODELPATH` environment variable. The environment variable supports paths that start with `~`, relative paths, and absolute paths. The path cannot contain symbolic links. When you use this variable, the path is converted to an absolute path and restricted to the `/home` or `/root` directory.

        ```bash
        mv op_models/* $PWD/../modelpath
        export MX_INDEX_MODELPATH=`realpath $PWD/../modelpath`
        ```

    - If you do not configure the path through an environment variable, move the operator model files to the `modelpath` directory in the current directory.

    After you generate the operators, store the relevant `.om` files properly and ensure that the files are not tampered with.

    > [!NOTE]
    >
    > If operator generation fails with `Failed to import Python module`, see [NumPy data type `np.float_` has been removed](./faq.md#numpy-data-type-npfloat_-has-been-removed) for a solution.

## Usage Example

This section provides a simple example to help users quickly try the retrieval flow with Index SDK.

Assume that on Atlas Inference Series products, a service uses the brute-force search (`Flat`) algorithm. The base library has 1 million vectors, the feature dimension is 512, the number of vectors to search is 128, and `topk` is 10. The general steps for writing a demo that calls the Index interface are as follows.

**Prerequisites**

- You have completed [Installation and Deployment](./installation_guide.md#installation-and-deployment).
- You have generated the [Flat](#generating-operators) and [AICPU](#generating-operators) operators.

**Procedure**

1. Build the demo. The process includes:

    1. Include the header file for brute-force search (`Flat`) in the demo.
    2. Construct the base library vector data. This example uses randomly generated data instead.
    3. Normalize the base library data.
    4. Initialize the Flat index.
    5. Call the interface to add the base library.
    6. Call the interface to run retrieval.

    The `demo.cpp` code is as follows:

    ```cpp
    #include <faiss/ascend/AscendIndexFlat.h>
    #include <sys/time.h>
    #include <random>
    // Get the current time.
    inline double GetMillisecs()
    {
        struct timeval tv = {0, 0};
        gettimeofday(&tv, nullptr);
        return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
    }
    // Build base library data with random numbers.
    void Generate(size_t ntotal, std::vector<float> &data, int seed = 5678)
    {
        std::default_random_engine e(seed);
        std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
        data.resize(ntotal);
        for (size_t i = 0; i < ntotal; ++i) {
            data[i] = static_cast<float>(255 * rCode(e) - 128);
        }
    }
    // Normalize base library data.
    void Norm(size_t total, std::vector<float> &data, int dim)
    {
        for (size_t i = 0; i < total; ++i) {
            float mod = 0;
            for (int j = 0; j < dim; ++j) {
                mod += data[i * dim + j] * data[i * dim + j];
            }
            mod = sqrt(mod);
            for (int j = 0; j < dim; ++j) {
                data[i * dim + j] = data[i * dim + j] / mod;
            }
        }
    }
    int main()
    {
        int dim = 512;
        std::vector<int> device{0};
        size_t ntotal = 1000000;
        int searchnum = 128;
        std::vector<float> features(dim * ntotal);
        int64_t resourceSize = static_cast<int64_t>(1024) * 1024 * 1024;
        int topK = 10;
        printf("Generating random numbers start!\r\n");
        Generate(ntotal, features);
        Norm(ntotal, features, dim);
        try {
            // Initialize the index.
            faiss::ascend::AscendIndexFlatConfig conf(device, resourceSize);
            auto metricType = faiss::METRIC_INNER_PRODUCT;
            faiss::ascend::AscendIndexFlat index(dim, metricType, conf);
            index.reset();
            // Add the base library.
            printf("add start!\r\n");
            index.add(ntotal, features.data());
            size_t tmpTotal = index.getBaseSize(0);
            if (tmpTotal != ntotal) {
                printf("------- Error -----------------\n");
                return -1;
            }
            // Search.
            printf("search start!\r\n");
            int loopTimes = 1;
            std::vector<float> dist(searchnum * topK, 0);
            std::vector<faiss::idx_t> label(searchnum * topK, 0);
            auto ts = GetMillisecs();
            for (int i = 0; i < loopTimes; i++) {
                index.search(searchnum, features.data(), topK, dist.data(), label.data());
            }
            auto te = GetMillisecs();
            printf("search end!\r\n");
            printf("flat, base:%lu, dim:%d, searchnum:%d, topk:%d, duration:%.3lf, QPS:%.4f\n",
                ntotal,
                dim,
                searchnum,
                topK,
                te - ts,
                1000 * searchnum * loopTimes / (te - ts));
            return 0;
        } catch(...) {
            printf("Exception caught! \r\n");
            return -1;
        }
    }
    ```

2. Compile `demo.cpp`.

    ```bash
    # Take /home/work/FeatureRetrieval as an example installation path.
    g++ --std=c++11 -fPIC -fPIE -fstack-protector-all -Wall -D_FORTIFY_SOURCE=2 -O3  -Wl,-z,relro,-z,now,-z,noexecstack -s -pie \
    -o demo demo.cpp \
    -I/home/work/FeatureRetrieval/mxIndex/include \
    -I/usr/local/faiss/faiss1.10.0/include \
    -I/usr/local/Ascend/driver/include \
    -I/opt/OpenBLAS/include \
    -L/home/work/FeatureRetrieval/mxIndex/host/lib \
    -L/usr/local/faiss/faiss1.10.0/lib \
    -L/usr/local/Ascend/driver/lib64 \
    -L/usr/local/Ascend/driver/lib64/driver \
    -L/opt/OpenBLAS/lib \
    -L/usr/local/Ascend/ascend-toolkit/latest/lib64 \
    -lfaiss -lascendfaiss -lopenblas -lc_sec -lascendcl -lascend_hal -lascendsearch -lock_hmm -lacl_op_compiler
    ```

3. Run the demo. If `search end!` appears, the demo ran successfully.

    ```bash
    ./demo
    ...
    search end!
    ```

## Algorithm Introduction

> [!NOTE]
>
> Standard deployment primarily uses AI CPUs. The recommended ratio of Ctrl CPUs to AI CPUs is as follows.
>
> - For Atlas 200/300/500 Inference Products, set it to 2:6.
> - For Atlas Inference Series products, set it to 1:7.

### Full Search

**Full Search Algorithm Introduction**

| Algorithm (API Reference) | Algorithm Usage Scenario | Operators to Generate | Sample Link |
| -- | -- | -- | -- |
| [AscendIndexInt8Flat](./api/full_retrieval.md#ascendindexint8flat) |<li>Feature type: int8</li><li>Feature dimension: 64, 128, 256, 384, 512, 768, 1024</li><li>Distance type: L2 and IP</li><li>Calculation precision: High</li><li>Device memory usage: Low</li><li>Applicable scenario: Brute-force search scenarios with high precision requirements</li> | <ul><li>[INT8Flat](#generating-operators)</li><li>[AICPU](#generating-operators)</li></ul> | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexInt8Flat.cpp) |
| [AscendIndexFlat](./api/full_retrieval.md#ascendindexflat) |<li>Feature type: FP32, FP16</li><li>Feature dimension: 32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096</li><li>Distance type: L2 and IP</li><li>Calculation precision: High</li><li>Device memory usage: High</li><li>Applicable scenario: Brute-force search scenarios with high precision requirements. IP distance is recommended when `dim > 128`.</li> | <ul><li>[Flat](#generating-operators)</li><li>[AICPU](#generating-operators)</li></ul> | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexFlat.cpp) |
| [AscendIndexSQ](./api/full_retrieval.md#ascendindexsq) |<li>Feature type: FP32</li><li>Feature dimension: 64, 128, 256, 384, 512, 768</li><li>Distance type: L2 and IP</li><li>Calculation precision: High</li><li>Device memory usage: Low, because it is quantized to int8</li><li>Applicable scenario: Brute-force search scenarios with relatively high precision requirements</li> | <ul><li>[SQ8](#generating-operators)</li><li>[AICPU](#generating-operators)</li></ul> | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexSQ.cpp) |
| [AscendIndexCluster](./api/full_retrieval.md#ascendindexcluster) |<li>Feature type: FP32</li><li>Feature dimension: 32, 64, 128, 256, 384, 512</li><li>Distance type: IP</li><li>Calculation precision: High</li><li>Device memory usage: Relatively high</li><li>Applicable scenario: Clustering scenarios that only calculate distance</li><li>Only supports Atlas Inference Series products</li>| <ul><li>[Flat](#generating-operators)</li><li>[AICPU](#generating-operators)</li></ul> | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexCluster.cpp) |
| [IndexIL](./api/full_retrieval.md#indexil) | It needs to run on the Device. Installation and deployment are complex, so it is not recommended for now. | - [Flat](#generating-operators);  | See [IndexILFlat](./api/full_retrieval.md#indexilflat) |
| [AscendIndexILFlat](./api/full_retrieval.md#ascendindexilflat) |<li>Feature type: FP16, FP32</li><li>Feature dimension: 32, 64, 128, 256, 384, 512</li><li>Distance type: IP</li><li>Calculation precision: High</li><li>Device memory usage: Relatively high</li><li>Applicable scenario: Clustering scenarios that only calculate distance</li><li>Only supports Atlas Inference Series products</li> | <ul><li>[Flat](#generating-operators)</li><li>[AICPU](#generating-operators)</li></ul> | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/tree/master/IndexSDK) |

### Approximate Search

**Approximate Search Algorithm Introduction**

| Algorithm (API Reference) | Algorithm Usage Scenario | Operators to Generate | Sample Link |
| -- | -- | -- | -- |
| [AscendIndexIVFSP](./api/approximate_retrieval.md#ascendindexivfsp) |<li>Feature type: FP32</li><li>Feature dimension: 64, 128, 256, 512, 768</li><li>Distance type: L2</li><li>Calculation precision: Medium</li><li>Device memory usage: Low, because features are compressed</li><li>Applicable scenario: Suitable for approximate search scenarios with billion-level base libraries, high performance requirements, and tolerance for precision loss.</li><li>Only supports Atlas Inference Series products</li>| <ul><li>IVFSP service operator</li><li>IVFSP AICPU operator</li><li>IVFSP training operator (used only when codebook files need to be generated through training)</li><li>See [IVFSP](#generating-operators).</li></ul> | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFSP.cpp) |
| [AscendIndexIVFSQ](./api/approximate_retrieval.md#ascendindexivfsq) |<li>Feature type: FP32</li><li>Feature dimension: 64, 128, 256, 384, 512</li><li>Distance type: L2 and IP</li><li>Calculation precision: Medium</li><li>Device memory usage: Low, because it is quantized to int8</li><li>Applicable scenario: The IVFSQ algorithm acts as a performance-precision tradeoff, suitable for scenarios that tolerate precision loss but require high performance.</li> | <ul><li>[IVFSQ8](#generating-operators)</li><li>[AICPU](#generating-operators)</li><li>[FlatAT](#generating-operators) (generate the FlatAT operator only when `useKmeansPP` is set to `true`.)</li></ul> | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFSQ.cpp) |
| [AscendIndexIVFSQT](./api/approximate_retrieval.md#ascendindexivfsqt) |<li>Feature type: FP32</li><li>Feature dimension: 256</li><li>Distance type: IP</li><li>Calculation precision: Medium</li><li>Device memory usage: Low, because of quantization and dimensionality reduction</li><li>Applicable scenario: AscendIndexIVFSQT is a three-stage IVFSQ retrieval algorithm that includes dimensionality reduction. It is suitable for approximate search scenarios with billion-level base libraries, high performance requirements, and tolerance for precision loss.</li> | <ul><li>[IVFSQT](#generating-operators)</li><li>[FlatAT](#generating-operators)</li><li>[AICPU](#generating-operators)</li><li>[FlatInt8AT](#generating-operators) (required on Atlas Inference Series products)</li></ul> | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFSQT.cpp) |
| [AscendIndexBinaryFlat](./api/approximate_retrieval.md#ascendindexbinaryflat) |<li>Feature type: uint8 binary features</li><li>Feature dimension: 256, 512, 1024</li><li>Distance type: Hamming and IP</li><li>Calculation precision: High</li><li>Device memory usage: Low</li><li>Applicable scenario: The AscendIndexBinaryFlat class inherits from Faiss `IndexBinary` and is used for binary feature retrieval. It suits scenarios with low memory usage requirements and high performance requirements.</li><li>Only supports Atlas Inference Series products</li> | <ul><li>[BinaryFlat](#generating-operators)</li><li>[AICPU](#generating-operators)</li></ul> | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexBinaryFlat.cpp) |
| [AscendIndexVStar](./api/approximate_retrieval.md#ascendindexvstar) |<li>Feature type: FP32</li><li>Feature dimension: 128, 256, 512, 1024</li><li>Distance type: L2</li><li>Calculation precision: Medium</li><li>Device memory usage: Low, because features are compressed</li><li>Applicable scenario: Suitable for approximate search scenarios with tens of millions of base vectors, high performance requirements, and tolerance for precision loss.</li><li>Only supports Atlas Inference Series products</li> | <ul><li>VStar service operator</li><li>VStar AICPU operator</li><li>VStar training operator (used only when codebook files need to be generated through training)</li><li>See [VSTAR](#generating-operators).</li></ul> | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexVStar.cpp) |
| [AscendIndexGreat](./api/approximate_retrieval.md#ascendindexgreat) |<li>Feature type: FP32</li><li>Feature dimension: 128, 256, 512, 1024</li><li>Distance type: L2</li><li>Calculation precision: Medium</li><li>Device memory usage: Low, because features are compressed</li><li>Applicable scenario: Suitable for approximate search scenarios with tens of millions of base vectors, high performance requirements, and tolerance for precision loss.</li><li>Only supports Atlas Inference Series products</li> | <ul><li>VStar service operator</li><li>VStar AICPU operator</li><li>VStar training operator (used only when codebook files need to be generated through training)</li><li>See [VSTAR](#generating-operators).</li></ul> | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexGreat.cpp) |
| [AscendIndexIVFFlat](./api/approximate_retrieval.md#ascendindexivfflat) |<li>Feature type: FP32</li><li>Feature dimension: 128</li><li>Distance type: IP</li><li>Calculation precision: Medium</li><li>Device memory usage: Medium</li><li>Applicable scenario: Suitable for approximate search scenarios with billion-level base libraries, high performance requirements, and tolerance for precision loss.</li><li>Only supports Atlas A2, A3, and Ascend950 inference series products</li> | <ul><li>[AICPU](#generating-operators)</li><li>[IVFFLAT](#ivfflat)</li></ul> | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFFlat.cpp) |
| [AscendIndexIVFPQ](./api/approximate_retrieval.md#ascendindexivfpq) |<li>Feature type: FP32</li><li>Feature dimension: 128</li><li>Distance type: IP</li><li>Calculation precision: Medium</li><li>Device memory usage: Medium</li><li>Applicable scenario: Suitable for approximate search scenarios with billion-level base libraries, high performance requirements, and tolerance for precision loss.</li><li>Only supports Atlas A2, A3, and Ascend950 inference series products</li> | <ul><li>[AICPU](#generating-operators)</li><li>[IVFPQ](#ivfpq)</li></ul> | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFPQ.cpp) |
| [AscendIndexIVFRaBitQ](./api/approximate_retrieval.md#ascendindexivfrabitq) |<li>Feature type: FP32</li><li>Feature dimension: 128</li><li>Distance type: L2 and IP</li><li>Calculation precision: Medium</li><li>Device memory usage: Low, because features are compressed</li><li>Applicable scenario: Suitable for approximate search scenarios with billion-level base libraries, high performance requirements, and tolerance for precision loss.</li><li>Only supports Atlas A2, A3, and Ascend950 inference series products</li> | <ul><li>[AICPU](#generating-operators)</li><li>[IVFRaBitQ](#ivfrabitq)</li></ul> | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFRabitQ.cpp) |

### Attribute Filter Search

**Attribute Filter Search Algorithm Introduction**

| Algorithm (API Reference) | Algorithm Usage Scenario | Operators to Generate | Sample Link |
| -- | -- | -- | -- |
| [AscendIndexTS](./api/attribute_filtering-based_retrieval.md#ascendindexts) |<li>Feature type: uint8 binary features, int8, FP32, depending on the algorithm</li><li>Feature dimension: depends on the specific algorithm</li><li>Distance type: Hamming, Cos, IP, L2</li><li>Calculation precision: Relatively high</li><li>Device memory usage: Relatively high</li><li>Applicable scenario: Spatial-temporal library scenarios that require attribute filtering</li><li>Hamming distance is supported only on Atlas Inference Series products</li> | <ul><li>[Mask](#generating-operators)</li><li>[BinaryFlat](#generating-operators)</li><li>[Int8Flat](#generating-operators)</li><li>[Flat](#generating-operators)</li><li>[AICPU](#generating-operators)</li></ul> | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexTS.cpp) |

### Multi-Index Batch Search

**Interface Introduction**

| Interface (API Reference) | Interface Usage Scenario | Algorithms That Can Use This Interface | Sample Link |
| -- | -- | -- | -- |
| [Search](./api/multi-index_batch_retrieval.md#search-faissindex) | Search on multiple indexes on a single Device. |<li>[AscendIndexSQ](./api/full_retrieval.md#ascendindexsq)</li><li>[AscendIndexFlat](./api/full_retrieval.md#ascendindexflat)</li><li>[AscendIndexIVFSP](./api/approximate_retrieval.md#ascendindexivfsp)</li>|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a>|
| [Search](./api/multi-index_batch_retrieval.md#search-ascendindex) | Search on multiple AscendIndex instances on a single Device. | <li>[AscendIndexSQ](./api/full_retrieval.md#ascendindexsq)</li><li>[AscendIndexFlat](./api/full_retrieval.md#ascendindexflat)</li><li>[AscendIndexIVFSP](./api/approximate_retrieval.md#ascendindexivfsp)</li>|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a>|
| [Search](./api/multi-index_batch_retrieval.md#search-ascendindexint8) | Search on multiple AscendIndexInt8 instances on a single Device.|<li>[AscendIndexInt8Flat](./api/full_retrieval.md#ascendindexint8flat)</li>|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a>|
| [SearchWithFilter](./api/multi-index_batch_retrieval.md#searchwithfilter-faissindex-single-filter) | Search on multiple indexes with attribute filtering on a single Device, single filter. |<li>[AscendIndexSQ](./api/full_retrieval.md#ascendindexsq)</li><li>[AscendIndexIVFSP](./api/approximate_retrieval.md#ascendindexivfsp)</li>|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a>|
| [SearchWithFilter](./api/multi-index_batch_retrieval.md#searchwithfilter-ascendindex-single-filter) | Search on multiple AscendIndex instances with attribute filtering on a single Device, single filter.|<li>[AscendIndexSQ](./api/full_retrieval.md#ascendindexsq)</li><li>[AscendIndexIVFSP](./api/approximate_retrieval.md#ascendindexivfsp)</li>|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a>|
| [SearchWithFilter](./api/multi-index_batch_retrieval.md#searchwithfilter-faissindex-multiple-filters) | Search on multiple indexes with filtering attributes on a single Device, multiple filters.|<li>[AscendIndexSQ](./api/full_retrieval.md#ascendindexsq)</li><li>[AscendIndexIVFSP](./api/approximate_retrieval.md#ascendindexivfsp)</li>|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a>|
| [SearchWithFilter](./api/multi-index_batch_retrieval.md#searchwithfilter-ascendindex-multiple-filters) | Search on multiple AscendIndex instances with attribute filtering on a single Device, multiple filters. |<li>[AscendIndexSQ](./api/full_retrieval.md#ascendindexsq)</li><li>[AscendIndexIVFSP](./api/approximate_retrieval.md#ascendindexivfsp)</li>|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a>|

### Other Functions

**Algorithm Introduction**

| Algorithm (API Reference) | Algorithm Requirements, Such as Performance and Scenario Differences | Invocation Method | Sample Link |
| -- | -- | -- | -- |
| [IReduction](./api/more_functions.md#ireduction) | IReduction is a unified interface for dimensionality reduction methods in the feature retrieval component. It currently supports the `PCAR` and `NN` dimensionality reduction algorithms. | Initialize it with `ReductionConfig`, call `CreateReduction` to create the reduction object, and then call `train` and `reduce`. | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIReduction.cpp) |
| [AscendNNInference](./api/more_functions.md#ascendnninference) | Perform inference through a neural network. | Create the NN reduction object with `AscendNNInference`, and then call `infer` for dimensionality reduction. | [Link](https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIReduction.cpp) |
| [AscendCloner](./api/more_functions.md#ascendcloner) | Index SDK provides a way to copy retrieval Index resources on the NPU to CPU-side Faiss. The copy process happens in memory, and the data loaded in the original NPU Index is copied to CPU memory so users can perform retrieval on the CPU with the same base library. | `index_ascend_to_cpu` copies an AscendIndex to a CPU Index. `index_cpu_to_ascend` copies a CPU Index to an AscendIndex. | None |

## Custom Operator Introduction

### Custom Operator Overview

The feature retrieval solution uses TIK operators to implement feature distance calculation logic. It includes the following custom operators.

- [Flat distance calculation operator](#generating-operators): Computes the distance between the feature base library data and the feature vector to be searched, for L2/IP.
- [SQ8 distance calculation operator](#generating-operators): Computes the distance between SQ-quantized base library data and the unquantized feature vector to be searched, for L2/IP.
- [IVFSQ8 operator](#generating-operators): Provides the operators required by the IVFSQ8 algorithm.
- [INT8Flat distance calculation operator](#generating-operators): Computes the distance between INT8-quantized base library data and the INT8-quantized feature vector to be searched, for L2/COS.
- [IVFSQT operator](#generating-operators): Provides the distance operators required for the three stages of IVFSQT.
- [FlatAT operator](#generating-operators): Mainly used in IVF scenarios to reduce the time consumed by train and add. Here, `code_num` is equal to `nlist`.
- [FlatInt8AT operator](#generating-operators): Optimizes the time consumed by train, add, and update in IVFSQT on Atlas Inference Series products.
- [AICPU operator](#generating-operators): Schedules the CPU on the Ascend AI Processor to perform sorting and other calculations, making full use of hardware performance.
- [BinaryFlat operator](#generating-operators): Provides the operators required by the binary algorithm.
- [Mask operator](#generating-operators): Provides the Mask operator required by the spatial-temporal library attribute filtering algorithm.
- [IVFSP operator](#generating-operators): Provides the service operator and AICPU operator required by the IVFSP algorithm, as well as the training operator used when generating the IVFSP codebook during training.
- [VStar operator](#generating-operators): Provides the service operator and AICPU operator required by the VStar algorithm.
- [IVFFLAT](#ivfflat): Provides the distance operators required by the first and second stages of IVFFLAT.
- [IVFPQ operator](#ivfpq): Provides the distance operators required by the first, second, and third stages of IVFPQ.
- [IVFRaBitQ operator](#ivfrabitq): Provides the operators required by IVFRaBitQ.

### Operator Generation Instructions

#### Flat

| Usage | `python3 flat_generate_model.py -d <dim> --cores <core_num> -p <process_id> -pool <pool_size> -t <npu_type>` |
| -- | -- |
| Parameters | <ul><li>`<dim>`: Feature vector dimension D. The default value is `512`.</li><li>`<core_num>`: Number of AI Cores on the Ascend AI Processor. The default value is `2`. No additional configuration is required.</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`<pool_size>`: Process pool size for multi-process scheduling during batch operator generation. The default value is `10`.</li><li>`<npu_type>`: Hardware form factor.</li><li>For Atlas 200/300/500 Inference Products and Atlas Inference Series products, run `npu-smi info` on the server where the Ascend AI Processor is installed. Remove the last digit from the reported `Name`, and the remaining value is `npu_type`.</li><li>For Atlas 800I A2 Inference Servers, run `npu-smi info` on the server where the Ascend AI Processor is installed. The reported `Name` is `npu_type`.</li><li>For Atlas 800I A3 Supernode Servers, run `npu-smi info -t board -i 0 -c 0` to obtain `NPU Name`. `910_` plus the `NPU Name` value is `npu_type`.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | Running this command generates a set of distance calculation operator model files. You need to modify the parameters in the command yourself. |
| Constraints | `dim` ∈ {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. 0 ≤ `pool_size` ≤ 32 |

**Involved Algorithms**

- [AscendIndexFlat](#full-search)
- [AscendIndexCluster](#full-search)
- [IndexIL](#full-search)
- [AscendIndexTS](#attribute-filter-search)
- [Search (multiple-index search on a single device)](./api/multi-index_batch_retrieval.md#search-faissindex)
- [Search (multiple AscendIndex searches on a single device)](./api/multi-index_batch_retrieval.md#search-ascendindex)

#### SQ8

> [!NOTE]
>
> The main difference between INT8Flat and SQ8 is that INT8 is quantized externally, and the Index input features are of the INT8 type. SQ8 is quantized internally by the Index, and the Index input features are of the Float32 type.

| Usage | `python3 sq8_generate_model.py -d <dim> --cores <core_num> -p <process_id> -pool <pool_size> -t <npu_type>` |
| -- | -- |
| Parameters | <ul><li>`<dim>`: Feature vector dimension D. The default value is `128`.</li><li>`<core_num>`: Number of AI Cores on the Ascend AI Processor. The default value is `2`. If this parameter is not specified, it is configured according to `<npu_type>`: when `npu_type` is `310`, `<core_num>` is `2`</li><li>when `npu_type` is `310P`, `<core_num>` is `8`.</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`<pool_size>`: Process pool size for multi-process scheduling during batch operator generation. The default value is `10`.</li><li>`<npu_type>`: Hardware form factor. Currently, `<npu_type>` supports Atlas 200/300/500 Inference Products and Atlas Inference Series products. The valid values are `310` and `310P`, and the default value is `310`.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | Running this command generates a set of SQ8 distance calculation operator model files. You need to modify the parameters in the command yourself. |
| Constraints | `dim` ∈ {64, 128, 256, 384, 512, 768}. 0 ≤ `pool_size` ≤ 32 |

**Involved Algorithms**

- [AscendIndexSQ](#full-search)
- [Search (multiple-index search on a single device)](./api/multi-index_batch_retrieval.md#search-faissindex)
- [Search (multiple AscendIndex searches on a single device)](./api/multi-index_batch_retrieval.md#search-ascendindex)
- [SearchWithFilter (FaissIndex single filter)](./api/multi-index_batch_retrieval.md#searchwithfilter-faissindex-single-filter)
- [SearchWithFilter (AscendIndex single filter)](./api/multi-index_batch_retrieval.md#searchwithfilter-ascendindex-single-filter)
- [SearchWithFilter (FaissIndex multiple filters)](./api/multi-index_batch_retrieval.md#searchwithfilter-faissindex-multiple-filters)
- [SearchWithFilter (AscendIndex multiple filters)](./api/multi-index_batch_retrieval.md#searchwithfilter-ascendindex-multiple-filters)

#### IVFSQ8

| Usage | `python3 ivfsq8_generate_model.py -d <dim> -c <coarse_centroid_num> --cores <core_num> -p <process_id> -pool <pool_size> -t <npu_type>` |
| -- | -- |
| Parameters | <ul><li>`<dim>`: Feature vector dimension D. The default value is `128`.</li><li>`<coarse_centroid_num>`: Number of L1 cluster centroids. The default value is `16384`.</li><li>`<core_num>`: Number of AI Cores on the Ascend AI Processor. The default value is `2`. If this parameter is not specified, it is configured according to `<npu_type>`: when `npu_type` is `310`, `<core_num>` is `2`</li><li>when `npu_type` is `310P`, `<core_num>` is `8`.</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`<pool_size>`: Process pool size for multi-process scheduling during batch operator generation. The default value is `10`.</li><li>`<npu_type>`: Hardware form factor. Currently, `<npu_type>` supports Atlas 200/300/500 Inference Products and Atlas Inference Series products. The valid values are `310` and `310P`, and the default value is `310`.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | Running this command generates a set of operator model files. You need to modify the parameters in the command yourself. |
| Constraints | `dim` ∈ {64, 128, 256, 384, 512}. `coarse_centroid_num` ∈ {1024, 2048, 4096, 8192, 16384, 32768}. 0 ≤ `pool_size` ≤ 32 |

**Involved Algorithms**

[AscendIndexIVFSQ](#approximate-search)

#### INT8Flat

> [!NOTE]
>
> The main difference between INT8Flat and SQ8 is that INT8 is quantized externally, and the Index input features are of the INT8 type. SQ8 is quantized internally by the Index, and the Index input features are of the Float32 type.

| Usage | `python3 int8flat_generate_model.py -d <dim> --cores <core_num> -p <process_id> -pool <pool_size> -t <npu_type> -code <code_num>` |
| -- | -- |
| Parameters | <ul><li>`<dim>`: Feature vector dimension D. The default value is `512`.</li><li>`<core_num>`: Number of AI Cores on the Ascend AI Processor. The default value is `2`. No additional configuration is required.</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`<pool_size>`: Process pool size for multi-process scheduling during batch operator generation. The default value is `10`.</li><li>`<npu_type>`: Hardware form factor. For Atlas 200/300/500 Inference Products and Atlas Inference Series products, run `npu-smi info` on the server where the Ascend AI Processor is installed. Remove the last digit from the reported `Name`, and the remaining value is `npu_type`. For Atlas 800I A2 Inference Servers, run `npu-smi info` on the server where the Ascend AI Processor is installed. The reported `Name` is `npu_type`.</li><li>`<code_num>`: The base library block size when the operator is called. The default value is `262144`. If it is not set, operators for all `code_num` values are generated by default.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | Running this command generates a set of operator model files. You need to modify the parameters in the command yourself. |
| Constraints | `dim` ∈ {64, 128, 256, 384, 512, 768, 1024}. 0 ≤ `pool_size` ≤ 32. `code_num` ∈ {16384, 32768, 65536, 131072, 262144} |

**Involved Algorithms**

- [AscendIndexInt8Flat](#full-search)
- [AscendIndexTS](#attribute-filter-search)
- [Search (single-device multiple AscendIndexInt8 searches)](./api/multi-index_batch_retrieval.md#search-ascendindexint8)

#### IVFSQT

> [!NOTE]
>
> To reduce the time consumed by train and add, you need to generate the FlatAT operator. The `dim` of Flat must be the same as the `dim_in` of IVFSQT, and the `code_num` of Flat must match the `coarse_centroid_num` of IVFSQT.

| Usage | `python3 ivfsqt_generate_model.py --cores <core_num> -d <dim_in> -r <compress_ratio> -c <coarse_centroid_num> -p <process_id> -pool <pool_size> -t <npu_type>` |
| -- | -- |
| Parameters | <ul><li>`<dim_in>`: Input feature vector dimension. The default value is `256`.</li><li>`<compress_ratio>`: Ratio of input to output dimensions. The default value is `4`. Range: `compress_ratio >= 1`.</li><li>`<coarse_centroid_num>`: Number of L1 cluster centroids. The default value is `16384`.</li><li>`<core_num>`: Number of AI Cores on the Ascend AI Processor. The default value is `2`. If this parameter is not specified, it is configured according to `<npu_type>`: when `npu_type` is `310`, `<core_num>` is `2`</li><li>when `npu_type` is `310P`, `<core_num>` is `8`.</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`<pool_size>`: Process pool size for multi-process scheduling during batch operator generation. The default value is `32`. Range: `1 <= pool_size <= 32`.</li><li>`<npu_type>`: Hardware form factor. Currently, `<npu_type>` supports Atlas 200/300/500 Inference Products and Atlas Inference Series products. The valid values are `310` and `310P`, and the default value is `310`.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | Running this command generates a set of operator model files. |
| Constraints | `<dim_in>` ∈ {256}. `<compress_ratio>` ∈ {2, 4, 8}. `<coarse_centroid_num>` ∈ {1024, 2048, 4096, 8192, 16384, 32768}. `<dim_in>` can be divisible by `<compress_ratio>`. |

**Involved Algorithms**

[AscendIndexIVFSQT](#approximate-search)

#### FlatAT

> [!NOTE]
>
> The current FlatAT operator is used together with IVF-type operators to speed up the `add` and `train` processes of IVF-type operators. You cannot call FlatAT directly. The current `add` and `train` acceleration feature is specified through `AscendIndexIVFConfig.useKmeansPP` in IVF. In this case, training is supported only when the training scale is less than 7,000,000.

| Usage | `python3 flat_at_generate_model.py --cores <core_num> -d <dim> -c <code_num> -p <process_id> -t <npu_type>` |
| -- | -- |
| Parameters | <ul><li>`<dim>`: Input feature vector dimension. The default value is `64`.</li><li>`<code_num>`: Number of base library features to compare with the input feature. The default value is `8192`.</li><li>`<core_num>`: Number of AI Cores on the Ascend AI Processor. The default value is `2`. If this parameter is not specified, it is configured according to `<npu_type>`: when `npu_type` is `310`, `<core_num>` is `2`</li><li>when `npu_type` is `310P`, `<core_num>` is `8`.</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`<npu_type>`: Hardware form factor. Currently, `<npu_type>` supports Atlas 200/300/500 Inference Products and Atlas Inference Series products. The valid values are `310` and `310P`, and the default value is `310`.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | Running this command generates a set of operator model files.; FlatAT operators are mainly used in IVF scenarios to reduce the time consumed by train and add. |
| Constraints | `dim` ∈ {64, 128, 256}. `code_num` ∈ {1024, 2048, 4096, 8192, 16384, 32768} |

**Involved Algorithms**

- [AscendIndexIVFSQ](#approximate-search)
- [AscendIndexIVFSQT](#approximate-search)

#### FlatInt8AT

| Usage | `python3 flat_at_int8_generate_model.py --cores <core_num> -d <dim> -c <code_num> -p <process_id> --soc-version <soc_version> -t <npu_type>` |
| -- | -- |
| Parameters | <ul><li>`<core_num>`: Number of AI Cores on the Ascend AI Processor. The default value is `8`.</li><li>`<dim>`: Input feature vector dimension. The default value is `256`.</li><li>`<code_num>`: Number of base library features to compare with the input feature. The default value is `16384`.</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`<soc_version>`: Model of the Ascend AI Processor. The default value is `Ascend310P3`. No additional configuration is required.</li><li>`<npu_type>`: Hardware form factor. Currently, only Atlas Inference Series products are supported. The default value is `310P`. No additional configuration is required.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | Running this command generates a set of operator model files.; FlatInt8AT optimizes the time consumed by train, add, and update in IVFSQT for Atlas Inference Series usage scenarios. |
| Constraints | `dim` ∈ {256}. `code_num` ∈ {1024, 2048, 4096, 8192, 16384, 32768}. `soc_version` ∈ {Ascend310P3} |

**Involved Algorithms**

[AscendIndexIVFSQT](#approximate-search)

#### AICPU

| Usage | `python3 aicpu_generate_model.py --cores <core_num> -p <process_id> -t <npu_type>` |
| -- | -- |
| Parameters | <ul><li>`<core_num>`: Number of AI Cores on the Ascend AI Processor. The default value is `2`. (Reserved parameter, not used at present.)</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`<npu_type>`: Hardware form factor. Currently, `<npu_type>` supports Atlas 200/300/500 Inference Products, Atlas Inference Series products, and Atlas A2 Inference Series products. The default value is `310`. If you cannot determine the exact `npu_type`, run `npu-smi info` on the server where the Ascend AI Processor is installed. Remove the last digit from the reported `Name`, and the remaining value is `npu_type`. For Atlas 800I A3 Supernode Servers, you can run `npu-smi info -t board -i 0 -c 0` to obtain `NPU Name`. `910_` plus the `NPU Name` value is `npu_type`.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | Running this command generates a set of operator model files.; AICPU operator model files only need to be generated once, and operators for all algorithms are generated. |

**Involved Algorithms**

- [AscendIndexInt8Flat](#full-search)
- [AscendIndexFlat](#full-search)
- [AscendIndexSQ](#full-search)
- [AscendIndexCluster](#full-search)
- [AscendIndexIVFSQ](#approximate-search)
- [AscendIndexBinaryFlat](#approximate-search)
- [AscendIndexTS](#attribute-filter-search)
- [AscendIndexIVFSQT](#approximate-search)
- [AscendIndexIVFFlat](#approximate-search)
- [AscendIndexIVFPQ](#approximate-search)
- [AscendIndexIVFRaBitQ](#approximate-search)

#### BinaryFlat

| Usage | `python3 binary_flat_generate_model.py -d <dim> -q <query_type> -p <process_id> -pool <pool_size>` |
| -- | -- |
| Parameters | <ul><li>`<dim>`: Dimension of the binary feature vector. `dim` ∈ {256, 512, 1024}. The default value is `512`.</li><li>`<query_type>`: Search type. The default is `uint8`. When you improve the performance of the `search` interface of the AscendIndexBinaryFlat algorithm, set it to `float`.</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`<pool_size>`: Process pool size for multi-process scheduling during batch operator generation. The default value is `16`.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | None. |

**Involved Algorithms**

- [AscendIndexBinaryFlat](#approximate-search)
- [AscendIndexTS](#attribute-filter-search)

#### Mask

| Usage | `python3 mask_generate_model.py -token <max_token_cnt> -p <process_id> -pool <pool_size> -t <npu_type>` |
| -- | -- |
| Parameters | <ul><li>`<max_token_cnt>`: Maximum number of tokens for operator generation. The default value is 2500. The recommended range is [1, 300000].</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`<pool_size>`: Process pool size for multi-process scheduling during batch operator generation. The default value is `16`.</li><li>`<npu_type>`: Hardware form factor. Only `310P` is supported.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | None. |

**Involved Interface**

[AscendIndexTS](#attribute-filter-search)

#### IVFSP

IVFSP search currently supports only the `310P` hardware form factor. It involves the following model file generation types:

- `ivfsp_generate_model.py`: Generates IVFSP service operator model files. For details, see [IVFSP service operator model file generation](#ivfsp-service-operator-model-file-generation).
- `ivfsp_aicpu_generate_model.py`: Generates IVFSP AICPU operator model files. For details, see [IVFSP AICPU operator model file generation](#ivfsp-aicpu-operator-model-file-generation).
- `ivfsp_generate_pyacl_model.py`: Generates the training operator model files required for IVFSP codebook training. For details, see [IVFSP training operator model file generation](#ivfsp-training-operator-model-file-generation).

##### IVFSP Service Operator Model File Generation

| Usage | `python3 ivfsp_generate_model.py --cores <core_num> -d <dim> -nonzero_num <low_dim> -nlist <k> -handle_batch <handle_batch> -code_num <code_num> -p <process_id> --pool <pool_size>` |
| -- | -- |
| Parameters | <ul><li>`<core_num>`: Number of AI Cores. The default value is `8`, and no additional configuration is required.</li><li>`<dim>`: Feature vector dimension. The default value is `256`.</li><li>`<low_dim>`: Number of non-zero dimensions after feature vector compression. The default value is `32`.</li><li>`<k>`: Number of cluster centroids. Keep this consistent with `<k>` in IVFSP training operator model file generation. The default value is `1024`.</li><li>`<handle_batch>`: Number of candidate buckets dispatched per calculation during retrieval. The default value is `32`.</li><li>`<code_num>`: Maximum number of samples per bucket dispatched per calculation during retrieval. If a bucket is too large, the program automatically splits it into multiple operator dispatches for distance calculation based on `code_num`. Keep this consistent with `<codebook_batch_size>` in IVFSP training operator model file generation. The default value is `32768`.</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`<pool_size>`: Process pool size for multi-process scheduling during batch operator generation. The default value is `16`.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | Running this command generates a set of AI Core operator model files for IVFSP retrieval. You need to modify the command parameters yourself. |
| Constraints | When `dim` ∈ {64, 128, 256}, `k` ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}. When `dim` ∈ {512, 768}, `k` ∈ {256, 512, 1024, 2048}. `low_dim` must be a multiple of 16 and less than or equal to `min(128, dim)`. `handle_batch` must be a multiple of 16, and `16 <= handle_batch <= 240`. `0 < pool_size <= 32`. |

##### IVFSP AICPU Operator Model File Generation

| Usage | `python3 ivfsp_aicpu_generate_model.py --cores <core_num> -p <process_id>` |
| -- | -- |
| Parameters | <ul><li>`<core_num>`: Number of AI Cores. The default value is `8`, and no additional configuration is required.</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | Running this command generates a set of IVFSP retrieval AICPU operator model files. |

##### IVFSP Training Operator Model File Generation

| Usage | `python3 ivfsp_generate_pyacl_model.py --cores <core_num> -d <dim> -nonzero_num <low_dim> -nlist <k> -batch_size <batch_size> -code_num <codebook_batch_size> -p <process_id>` |
| -- | -- |
| Parameters | <ul><li>`<core_num>`: Number of AI Cores. The default value is `8`, and no additional configuration is required.</li><li>`<dim>`: Feature vector dimension. The default value is `256`.</li><li>`<low_dim>`: Number of non-zero dimensions after feature vector compression. The default value is `32`.</li><li>`<k>`: Number of cluster centroids. Keep this consistent with `<k>` in IVFSP service operator model file generation. The default value is `1024`.</li><li>`<batch_size>`: Batch size used during training. The default value is `32768`.</li><li>`<codebook_batch_size>`: Maximum number of samples used to operate on the codebook each time during training. It must be a power of 2. Keep this consistent with `<code_num>` in IVFSP service operator model file generation. The default value is `32768`.</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | Running this command generates a set of operator model files for IVFSP retrieval. You need to modify the command parameters yourself. The generated IVFSP training operator model files are saved in the `op_models_pyacl` subdirectory of the current directory. |
| Constraints | When `dim` ∈ {64, 128, 256}, `k` ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}. When `dim` ∈ {512, 768}, `k` ∈ {256, 512, 1024, 2048}. `low_dim` must be a multiple of 16 and less than or equal to `min(128, dim)`. `batch_size` must be a multiple of 16. `codebook_batch_size` must be a multiple of 16. |

#### VSTAR

VSTAR search currently supports only Atlas Inference Series products. It involves generating the VSTAR service operator model file (`vstar_generate_models.py`). For details, see [VSTAR](#generating-operators).

The operator generation environment must match the codebook generation environment. For details, see [Overall Description](#overall-description).

##### VSTAR Service Operator Model File Generation

| Usage | `python3 vstar_generate_models.py --dim <dim> --nlistL1 <nlist1> --subDimL1 <sub_dim1> --nProbeL1 <nprobe1> --nProbeL2 <nprobe2> --segmentNumL3 <segment> --pool <pool_size>` |
| -- | -- |
| Parameters | <ul><li>`<dim>`: Feature vector dimension. The default value is `256`.</li><li>`<nlist1>`: Number of first-level cluster centroids. The default value is `1024`.</li><li>`<nprobe1>`: Number of first-level candidate buckets dispatched for each retrieval calculation. The default value is `[72]`.</li><li>`<nprobe2>`: Number of second-level candidate buckets dispatched for each retrieval calculation. The default value is `[64, 296]`.</li><li>`<sub_dim1>`: Dimensionality after first-level reduction during retrieval. The default value is `32`.</li><li>`<segment>`: Number of data segments searched from `nprobe2` during retrieval. The default value is `[512, 1000, 1504]`.</li><li>`<pool_size>`: Process pool size for multi-process scheduling during batch operator generation. The default value is `16`.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | Running this command generates a set of AI Core and AICPU operator model files for VSTAR retrieval. You need to modify the command parameters yourself. |
| Constraints | `dim` ∈ {128, 256, 512, 1024}. `nlist1` ∈ {256, 512, 1024}. `sub_dim1` ∈ {32, 64, 128}. `sub_dim1` must be less than `dim`. `nprobe1` ∈ (16, `nlist1`]. `nprobe1` is a list of int values, and each value in the list must be a multiple of 8. `nprobe2` ∈ (16, `nprobe1 * n`]. When `dim` is 1024, `n` is 16. For other dimensions, `n` is 32. `nprobe2` is a list of int values, and each value in the list must be a multiple of 8. `segment` ∈ (100, 5000]. `segment` is a list of int values, and each value must be a multiple of 8. `pool_size` ∈ [1, 32]. Before you run the script, determine the maximum number of processes supported by the host machine and set it appropriately. |

**Involved Algorithms**

[AscendIndexVStar](./api/approximate_retrieval.md#ascendindexvstar)

[AscendIndexGreat](./api/approximate_retrieval.md#ascendindexgreat)

#### IVFFLAT

| Usage | `python3 ivfflat_generate_model.py -d <dim> -c <coarse_centroid_num> --cores <core_num> -p <process_id> -pool <pool_size> -t <npu_type>` |
| -- | -- |
| Parameters | <ul><li>`<dim>`: Feature vector dimension. The default value is `128`.</li><li>`<coarse_centroid_num>`: Number of first-level cluster centroids. The default value is `1024`.</li><li>`<core_num>`: Number of AI Cores on the Ascend AI Processor. The default value is `40`. If this parameter is not specified, it is configured according to `<npu_type>`: when `<npu_type>` is `910B3`, `<core_num>` is `40`.</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`<pool_size>`: Process pool size for multi-process scheduling during batch operator generation. The default value is `10`.</li><li>`<npu_type>`: Hardware form factor. Currently, `<npu_type>` supports Atlas A2, A3, and Ascend950 inference series products. The default value is `910B4`. If you cannot determine the specific `npu_type`, run `npu-smi info` on the server where the Ascend AI Processor is installed. Remove the last digit from the reported `Name`, and the remaining value is `npu_type`. For Atlas 800I A3 Supernode Servers, run `npu-smi info -t board -i 0 -c 0` to obtain `NPU Name`. `910_` plus the `NPU Name` value is `npu_type`. For Atlas Ascend950 Supernode Servers, set `npu_type` to `Ascend950PR`.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | Running this command generates a set of operator model files. You need to modify the parameters in the command yourself. |
| Constraints | `dim` ∈ {64, 128, 256, 384, 512}. `coarse_centroid_num` ∈ {1024, 2048, 4096, 8192, 16384, 32768}. 0 ≤ `pool_size` ≤ 32 |

**Involved Algorithms**

[AscendIndexIVFFlat](./api/approximate_retrieval.md#ascendindexivfflat)

#### IVFPQ

| Usage | `python3 ivfpq_generate_model.py -d <dim> -c <nlist> --cores <core_num> -m <m> -n <nbit> -topK <topK> -b <blockNum> -p <process_id> -t <npu_type>` |
| -- | -- |
| Parameters | <ul><li>`<dim>`: Feature vector dimension. The default value is `128`.</li><li>`<nlist>`: Number of first-level cluster centroids. The default value is `1024`.</li><li>`<core_num>`: Number of AI Cores on the Ascend AI Processor. The default value is `40`. If this parameter is not specified, it is configured according to `<npu_type>`.</li><li>`<m>`: Number of subspaces. The default value is `4`.</li><li>`<nbit>`: Number of bits in the quantization centroid for each subspace. The default value is `8`, and no additional configuration is required. It also determines the number of codebook centroids, `ksub = 1 << nbit`. When `nbit` is 8, `ksub` is 256.</li><li>`<topK>`: Number of nearest candidate vectors returned for each query vector. The default value is `320`, and no additional configuration is required.</li><li>`<blockNum>`: Number of candidate vector blocks to process. The default value is `128`, and no additional configuration is required.</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`<npu_type>`: Hardware form factor. The current default value is `Ascend950PR`. If you cannot determine the specific `npu_type`, run `npu-smi info` on the server where the Ascend AI Processor is installed. Remove the last digit from the reported `Name`, and the remaining value is `npu_type`. For Atlas 800I A3 Supernode Servers, run `npu-smi info -t board -i 0 -c 0` to obtain `NPU Name`. `910_` plus the `NPU Name` value is `npu_type`.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | Running this command generates a set of operator model files. You need to modify the parameters in the command yourself. |
| Constraints | `dim` ∈ {128}. `nlist` ∈ {1024, 2048, 4096, 8192, 16384}. `m` ∈ {2, 4, 8, 16}. `n` ∈ {8}. |

**Involved Algorithms**

[AscendIndexIVFPQ](./api/approximate_retrieval.md#ascendindexivfpq)

#### IVFRaBitQ

| Usage | `python3 ivfrabitq_generate_model.py -d <dim> -c <coarse_centroid_num> --cores <core_num> -p <process_id> -pool <pool_size> -t <npu_type> -m <metric_type>` |
| -- | -- |
| Parameters | <ul><li>`<dim>`: Feature vector dimension. The default value is `128`.</li><li>`<coarse_centroid_num>`: Number of first-level cluster centroids. The default value is `16384`.</li><li>`<core_num>`: Number of AI Cores on the Ascend AI Processor. The default value is `40`. If this parameter is not specified, it is configured according to `<npu_type>`: when `<npu_type>` is `910B3`, `<core_num>` is `40`.</li><li>`<process_id>`: Process ID for multi-process scheduling during batch operator generation. The default value is `0`. No additional configuration is required.</li><li>`<pool_size>`: Process pool size for multi-process scheduling during batch operator generation. The default value is `10`.</li><li>`<npu_type>`: Hardware form factor. The default value is `910B4`. If you cannot determine the specific `npu_type`, run `npu-smi info` on the server where the Ascend AI Processor is installed. Remove the last digit from the reported `Name`, and the remaining value is `npu_type`. For Atlas 800I A3 Supernode Servers, run `npu-smi info -t board -i 0 -c 0` to obtain `NPU Name`. `910_` plus the `NPU Name` value is `npu_type`.</li><li>`<metric_type>`: Vector calculation mode. Use this to explicitly specify whether to calculate using `L2` or `IP` distance. The default value is `L2`.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Description | Running this command generates a set of operator model files. You need to modify the parameters in the command yourself. |
| Constraints | `dim` ∈ {128}. `coarse_centroid_num` ∈ {1024, 2048, 4096, 8192, 16384, 32768}. 0 ≤ `pool_size` ≤ 32 |

**Involved Algorithms**

[AscendIndexIVFRaBitQ](./api/approximate_retrieval.md#ascendindexivfrabitq)

#### VSTAR Codebook File Generation

##### Overall Description

**Environment Setup**

The environment dependencies are as follows:

- `nnae` (`8.0.0 <= version < 8.5.0`)
- `python` (`version >= 3.9`)
- `torch` (`version >= 2.0.1`)
- `torch_npu` (`version >= 2.0.1.post4`)

- `numpy` (`version >= 1.26.4`)
- `scikit-learn` (`version >= 1.4.1.post1`)
- `tqdm` (`version >= 4.66.1`)

You can install `torch`, `torch_npu`, `numpy`, `scikit-learn`, and `tqdm` with `pip install`. Example:

```bash
pip install numpy tqdm scikit-learn torch_npu torch
```

Versions earlier than CANN 8.5.0 require a separate `nnae` installation. Follow these steps:

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

**Notes**

- If you see the following error when you import `torch` and `torch_npu`:

    ```text
    .../libgomp.so: cannot allocate memory in static TLS block
    ```

    run `export LD_PRELOAD=.../libgomp.so` using the `libgomp.so` path shown in the error message.

- If `numpy` installation fails because `pip` cannot install the following dependencies:

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

    run the following command.

    ```bash
    pip install attrs cloudpickle decorator jinja2 ml-dtypes psutil scipy tornado absl-py
    ```

- If you encounter the following issue while training the codebook:

    ```text
    OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata.
    Segmentation fault (core dumped)
    ```

    run:

    ```bash
    export OPENBLAS_NUM_THREADS=1
    ```

    This environment variable may affect performance. After codebook training finishes, you are advised to restore it to the preset value.

- Details about the `--useOfflineCompile` option:

    Online operator compilation takes longer than offline operator compilation. The `--useOfflineCompile` option controls whether offline operator compilation is used, which runs with a precompiled offline operator package. This method requires you to install the single-operator package in advance. The installation steps are as follows:

    1. Download the [operator package](https://www.hiascend.com/developer/download/community/result?module=cann&product=2&model=17).
    2. Run the following command to add execute permissions.
        - Versions earlier than CANN 8.5.0.

            ```bash
            chmod u+x ./Ascend-cann-kernels-{chip_type}_{version}_linux-{arch}.run
            ```

        - CANN 8.5.0 and later.

            ```bash
            chmod u+x ./Ascend-cann-{chip_type}-ops_{version}_linux-{arch}.run
            ```

    3. Run the following command to install it.
        - Versions earlier than CANN 8.5.0.

            ```bash
            ./Ascend-cann-kernels-{chip_type}_{version}_linux-{arch}.run --install
            ```

        - CANN 8.5.0 and later.

            ```bash
            ./Ascend-cann-{chip_type}-ops_{version}_linux-{arch}.run --install
            ```

    4. Set the environment variables according to the installation prompts.
        - Versions earlier than CANN 8.5.0.

            ```bash
            source /{kernels_installation_path}/kernels/set_env.sh
            ```

        - CANN 8.5.0 and later.

            ```bash
            source /usr/local/Ascend/cann/set_env.sh
            ```

##### Codebook Training Script

Training uses the `vstar_train_codebook.py` script. The training script is in the `tools/train` folder under the installation directory. Note that the Python version is 3.9.

| Command reference | `python3 vstar_train_codebook.py --dataPath <data_path> --dim <dim> --codebookPath <codebook_output_dir> --nlistL1 <nlist1> --subDimL1 <sub_dim1> --device <device> --batchSize <batch_size> --sample <sample> --useOfflineCompile` |
| -- | -- |
| Parameters | <ul><li>`<data_path>`: Path to the raw data used for codebook training. The data must exist. This parameter is required.</li><li>`<dim>`: Feature vector dimension. Keep it consistent with the `<dim>` used when generating the VSTAR training operator model file. The default value is `256`.</li><li>`<codebook_output_dir>`: Path that stores the final codebook file. Ensure that the directory exists and that the user running the program has write permission. For security hardening reasons, the directory tree cannot contain symbolic links.</li><li>`<nlist1>`: Number of first-level cluster centroids. Keep it consistent with `<nlist1>` in VSTAR training operator model file generation. The default value is `1024`.</li><li>`<sub_dim1>`: Dimensionality after first-level reduction during retrieval. Keep it consistent with `<sub_dim1>` in VSTAR training operator model file generation. The default value is `32`.</li><li>`<device>`: Device logical ID. Run training on the specified Device. The default value is `1`.</li><li>`<batch_size>`: Batch size used during training. Range: `(0, 10240]`. The default value is `10240`.</li><li>`<sample>`: Sampling rate for the raw samples used in training. Range: `0 < ratio <= 1.0`. The default value is `1.0`.</li><li>`--useOfflineCompile`: Controls whether to use the operator package dependency and run offline operator compilation to improve performance. Disabled by default. If you enable it, add this option to the end of the command line. For details, see the section on the `--useOfflineCompile` option in the VSTAR codebook generation overview.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Usage instructions | The size of the raw data in `<data_path>` must be less than or equal to 10 million 1024-dimensional vectors, that is, `10,000,000 * 1024 * 4 = 40,960,000,000`. Running this command generates a new `codebook_<dim>_<nlist1>_<sub_dim1>.bin` directory under `<codebook_output_dir>`. This is the codebook file required by `AscendIndexVStar` and `AscendIndexGreat`. If the codebook file already exists, it is overwritten. In that case, the user running the program should be the file owner. Before you train and generate the codebook, first refer to VSTAR and generate the training operator model files. |

#### (Optional) Generate Codebook Files in Python

##### IVFSP Training Script

**Environment Setup**

The environment dependencies are as follows:

- `numpy` (`version > 1.16.0`)
- `tqdm` (`version >= 4.65.0`)
- `faiss-cpu` (`version = 1.10.0`)

You can install them with `pip install` as follows.

```bash
pip install numpy tqdm faiss-cpu==1.10.0
```

Before you run the training script, run the following command to set the environment variables.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**Run the Training Script**

Index SDK provides two ways to use the training script:

- Use the `trainCodeBook` interface of the IVFSP algorithm for training. This is the recommended method.
- Use the `ivfsp_train_codebook.py` script for training. The script is in the `tools/train` folder under the installation directory. Note that the Python version is 3.9.11. For convenience, an `ivfsp_train_codebook_example.sh` sample script is provided in the same folder. You can modify the parameters in that file according to your actual scenario and then run the script to generate the codebook file.

| Command reference | `python3 ivfsp_train_codebook.py --dim <dim> --nonzero_num <nonzero_num> --nlist <nlist> --num_iter <num_iter> --device <device> --batch_size <batch_size> --code_num <code_num> --ratio <ratio> --learn_data_path <learn_data_path> --codebook_output_dir <codebook_output_dir> --train_model_dir <train_model_dir>` |
| -- | -- |
| Parameters | <ul><li>`<dim>`: Feature vector dimension. Keep it consistent with the `<dim>` used when generating the IVFSP training operator model file. It must be greater than 0.</li><li>`<nonzero_num>`: Number of non-zero dimensions after feature vector compression. Keep it consistent with `<low_dim>` used when generating the IVFSP training operator model file. It must be greater than 0.</li><li>`<nlist>`: Number of cluster centroids. Keep it consistent with `<k>` used when generating the IVFSP training operator model file. It must be greater than 0.</li><li>`<num_iter>`: Training iteration count. The default value is 20. Setting it too large increases the training time. It must be greater than 0.</li><li>`<device>`: Device logical ID. Run training on the specified Device. The default value is `0`.</li><li>`<batch_size>`: Batch size used during training. Keep it consistent with `<batch_size>` used when generating the IVFSP training operator model file. It must be greater than 0 and less than or equal to 32768. The default value is `32768`.</li><li>`<code_num>`: The maximum number of samples used to operate on the codebook each time. It must be a power of 2. Keep it consistent with `<codebook_batch_size>` used when generating the IVFSP training operator model file. It must be greater than 0 and less than or equal to 32768. The default value is `32768`.</li><li>`<ratio>`: Sampling rate for the raw samples used in training. Range: `0 < ratio <= 1.0`. The default value is `1.0`.</li><li>`<learn_data_path>`: Path to the raw feature file used for training. `bin` and `npy` formats are supported. In `bin` format, data is stored in row-major order and uses the `float32` data type.</li><li>`<codebook_output_dir>`: Directory where the generated codebook file is output. Ensure that the directory exists and that the user running the program has write permission. For security hardening, the directory tree cannot contain symbolic links.</li><li>`<train_model_dir>`: Directory that contains the IVFSP training operator model files.</li><li>`--help \| -h`: Query help information.</li></ul> |
| Usage instructions | Running this command generates `codebook_<dim>_<nonzero_num>_<nlist>.bin` and `codebook_<dim>_<nonzero_num>_<nlist>.npy` in the directory corresponding to `<codebook_output_dir>`. The `codebook_<dim>_<nonzero_num>_<nlist>.bin` file is the codebook file required by `AscendIndexIVFSP`. If the codebook file already exists, it is overwritten. In that case, the user running the program should be the file owner. Before you train and generate the codebook, first generate the training operator model files by following the instructions for IVFSP training operator model file generation. The size of the data specified by `learn_data_path` must be greater than or equal to `nonzero_num * nlist * sizeof(float32)` bytes. |

##### Dimensionality Reduction Training Script

**Environment Dependencies**

- Install Python 3.9. Python 3.9, Python 3.10, and Python 3.11 are supported, but Python 3.9 is recommended.
- Install Faiss 1.10.0. You can install it with `pip install` as follows.

    ```bash
    pip install faiss-cpu==1.10.0
    ```

- Install `torch_cpu` and `torch_npu`. For the installation method, see the [link](https://gitee.com/ascend/pytorch). Choose the matching version according to the version compatibility table.

**Train the Model**

The default path of the scripts in this section is `tools/train/reduction`.

1. Train the model.

    ```bash
    python3 call_train.py --dataset_dir=Dataset_Dir --val_dataset_dir=./valid --generate_val=True --save_path=./modelsDr --dim=512 --npu=0 --ratio=4 --metric=L2 --mode=train --train_size=100000 --epochs=20 --train_batch_size=8192 --infer_batch_size=128 --learning_rate=0.0005 --log_stride=500 --construct_neighbors=100 --queries_validation=1000
    ```

    | Parameters | Description |
    | -- | -- |
    | dataset_dir | Dataset path, string type, required. The current implementation reads `base.npy`, `query.npy`, and `gt.npy` by default. If your dataset uses different file names, you can implement your own dataset loading and modify the line that calls `get_train_data` in this script. For example, the original code is:<br>```# load dataset demo before training, modify here if you want to load your own dataset        #####################################################################        learn, base = get_train_data(args.dataset_dir, args.train_size)        #####################################################################```<br>You can change it to:<br>```# load dataset demo before training, modify here if you want to load your own dataset        #####################################################################        # learn, base = get_train_data(args.dataset_dir, args.train_size)        learn = np.fromfile(YOUR_LEARN_DATASET_DIR, dtype=np.float32).reshape((-1, YOUR_DATA_DIM))        base = np.fromfile(YOUR_BASE_DATASET_DIR, dtype=np.float32).reshape((-1, YOUR_DATA_DIM))        #####################################################################``` |
    | val_dataset_dir | Valid when `generate_val` is `True`. Path where the validation set is stored. String type. The default value is `./validation/`. |
    | generate_val | Whether to generate the validation set. Set it to `True` for the first training run. Bool type. The default value is `False`. |
    | save_path | Path where the model is stored. String type. Required. |
    | dim | Optional. Dataset dimension. Range: `[96, 128, 200, 256, 512, 2048]`. Int type. The default value is `512`. |
    | npu | Device ID used for training, that is, the device number. Int type. Only single-card training is supported. CPU training is used by default. |
    | ratio | Optional. Dimensionality reduction ratio. Range: `[2, 4, 8, 16]`. Int type. The default value is `8`. |
    | metric | Distance metric used when training the model. Optional values are `L2` and `IP`. String type. The default value is `L2`. |
    | mode | Optional. Range: [`train`, `infer`, `test`]. Currently, only `train` is supported. The default value is `train`, so no change is required. |
    | train_size | Training set size. The value must be smaller than the total number of samples in the entire dataset. It is used to randomly sample part of the data when loading the dataset for training. Int type. If you implement your own dataset loading, sample according to `train_size` to prevent training from taking too long. The default value is `100000`, and the value must be greater than 0 if you change it. |
    | epochs | Number of training epochs. Int type. Setting it too large significantly increases training time. The default value is `30`, and the value must be greater than 0 if you change it. |
    | train_batch_size | Batch size during training. The default value is `8192`. Int type. The value must be greater than 0 if you change it. |
    | infer_batch_size | Batch size during inference. The default value is `128`. Int type. The value must be greater than 0 if you change it. |
    | learning_rate | Learning rate. The default value is `0.0005`. Float type. The value must be greater than 0 if you change it. |
    | log_stride | Training log printing interval, in steps. The default value is `500`. Int type. The value must be greater than 0 if you change it. |
    | construct_neighbors | Range of nearest neighbors used when building the training set. It is used to build the special training set structure required for dimensionality reduction. The default value is `100`. Adjust it according to the number of face images corresponding to each person in the dataset. Int type. The value must be greater than 0 if you change it. |
    | queries_validation | Number of query vectors required to build the validation set. Int type. The default value is `1000`, and the value must be greater than 0 if you change it. |
    | --help \| -h | Query help information. |

2. Generate the OM model.

    Before you run the training script, run the following command to set the environment variables. Modify the paths according to the actual installation path of the CANN package.

    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
    ```

    1. Generate the OM model with precision 32.

        ```bash
        bash atc.sh {save_path} {om_name} {input_shape}
        ```

    2. Generate the OM model with precision 16.

        ```bash
        bash atc_16.sh {save_path} {om_name} {input_shape}
        ```

    - `{save_path}`: Required. Indicates the path where the model is stored. The file name in the path must end with `.onnx` or `.pb`. Otherwise, the script reads environment variable values such as `framework` and `input_format`, which causes the script to fail.
    - `{om_name}`: Optional. Indicates the name of the generated OM model. The default is the same as the ONNX model name.
    - `{input_shape}`: Optional. The default is the input dimension of the ONNX model, in the format `actual_input_1:infer_batch_size,dim`. The default value is recommended, and changing it is not recommended.
    - `bash atc.sh` and `bash atc_16.sh` support only Atlas Inference Series products.
