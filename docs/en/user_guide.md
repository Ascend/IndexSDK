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

    <table><tbody>
    <tr><td width="180" align="center" valign="middle"><strong>Directory or File</strong></td><td align="center" valign="middle"><strong>Description</strong></td></tr>
    <tr><td width="180" valign="middle">device</td><td valign="middle">Contains the dynamic libraries and header files for the IndexIL algorithm.</td></tr>
    <tr><td width="180" valign="middle">filelist.txt</td><td valign="middle">Package file list.</td></tr>
    <tr><td width="180" valign="middle">host</td><td valign="middle">Search dynamic library. When you perform feature search, link to the dynamic libraries in this folder.</td></tr>
    <tr><td width="180" valign="middle">include</td><td valign="middle">API header files.</td></tr>
    <tr><td width="180" valign="middle">lib</td><td valign="middle">Search dynamic library, linked to <code>host/lib</code>.</td></tr>
    <tr><td width="180" valign="middle">modelpath</td><td valign="middle">Directory for operator <code>.om</code> files. After the operators are compiled, place the <code>.om</code> files in this folder.</td></tr>
    <tr><td width="180" valign="middle">ops</td><td valign="middle">Contains the <code>custom_opp_&lt;arch&gt;.run</code> script for installing search algorithm operators.</td></tr>
    <tr><td width="180" valign="middle">script</td><td valign="middle">Contains the uninstall script <code>uninstall.sh</code> for uninstalling the Index SDK package.</td></tr>
    <tr><td width="180" valign="middle">tools</td><td valign="middle">Contains the Python scripts for operator generation.</td></tr>
    <tr><td width="180" valign="middle">version.info</td><td valign="middle">Contains version-related information.</td></tr>
    </tbody></table>

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

    <table><tbody>
    <tr><td width="140" align="center" valign="middle"><strong>Parameters</strong></td><td align="center" valign="middle"><strong>Description</strong></td></tr>
    <tr><td width="140" valign="middle">--help | -h</td><td valign="middle">Query help information.</td></tr>
    <tr><td width="140" valign="middle">--info</td><td valign="middle">Query package build information.</td></tr>
    <tr><td width="140" valign="middle">--list</td><td valign="middle">Query the file list.</td></tr>
    <tr><td width="140" valign="middle">--check</td><td valign="middle">Query package integrity.</td></tr>
    <tr><td width="140" valign="middle">--quiet |-q</td><td valign="middle">Optional parameter that enables silent installation. It reduces interactive output.</td></tr>
    <tr><td width="140" valign="middle">--nox11</td><td valign="middle">Deprecated interface with no practical effect.</td></tr>
    <tr><td width="140" valign="middle">--noexec</td><td valign="middle">Extract the package to the current directory without running the installation script. Use it with <code>--extract=&lt;path&gt;</code>, in the format <code>--noexec --extract=&lt;path&gt;</code>.</td></tr>
    <tr><td width="140" valign="middle">--extract=&lt;path&gt;</td><td valign="middle">Extract the files in the package to the specified directory. You can use it with <code>--noexec</code>.</td></tr>
    <tr><td width="140" valign="middle">--tar arg1 [arg2 ...]</td><td valign="middle">Run the tar command on the package and use the parameters after tar as command arguments. For example, <code>--tar xvf</code> extracts the contents of the run installer package to the current directory.</td></tr>
    </tbody></table>

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

        <table><tbody>
        <tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 run_generate_model.py -m &lt;mode&gt; -t &lt;npu_type&gt; -p &lt;pipeline&gt; -pool &lt;pool_size&gt;</code></strong></td></tr>
        <tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;mode&gt;</code>: Algorithm mode. <code>&lt;mode&gt;</code> supports <code>ALL</code> and one or more of <code>Flat</code>, <code>SQ8</code>, <code>IVFSQ8</code>, and <code>INT8</code>. Separate multiple values with commas, for example: <code>python3 run_generate_model.py -m Flat,IVFSQ8</code>. All algorithms are selected by default, so you can run <code>python3 run_generate_model.py</code> directly.<br>● <code>&lt;npu_type&gt;</code>: The chip name. - For Atlas 200/300/500 Inference Products and Atlas Inference Series products, run <code>npu-smi info</code> on the server where the Ascend AI Processor is installed. Remove the last digit from the reported <code>Name</code>, and the remaining value is <code>npu_type</code>.<br>● For Atlas 800I A2 Inference Servers, run <code>npu-smi info</code> on the server where the Ascend AI Processor is installed. The reported <code>Name</code> is <code>npu_type</code>.<br>● For Atlas 800I A3 Supernode Servers, run <code>npu-smi info -t board -i 0 -c 0</code> to obtain <code>NPU Name</code>. <code>910_</code> plus the <code>NPU Name</code> value is <code>npu_type</code>.<br>● <code>&lt;pipeline&gt;</code>: Whether to use multi-threaded parallel pipelines to generate operator models. The default value is <code>true</code>. When set to <code>true</code>, the default <code>pool_size</code> is <code>32</code>.<br>● <code>&lt;pool_size&gt;</code>: The process pool size for multi-process scheduling during batch operator generation.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
        <tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">● Running this command generates multiple sets of operator model files.<br>● Before you run it, update the <code>para_table.xml</code> file in the current directory and fill in the required parameters in the table.<br>● <code>1 ≤ pool_size ≤ 32</code>.</td></tr>
        </tbody></table>

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

<table><tbody>
<tr><td align="center" valign="middle" width="250"><strong>Algorithm (API Reference)</strong></td><td align="center" valign="middle"><strong>Algorithm Usage Scenario</strong></td><td align="center" valign="middle"><strong>Operators to Generate</strong></td><td valign="middle" width="140" align="center"><strong>Sample Link</strong></td></tr>
<tr><td valign="middle" width="250"><a href="./api/full_retrieval.md#ascendindexint8flat">AscendIndexInt8Flat</a></td><td valign="middle">● Feature type: int8<br>● Feature dimension: 64, 128, 256, 384, 512, 768, 1024<br>● Distance type: L2 and IP<br>● Calculation precision: High<br>● Device memory usage: Low<br>● Applicable scenario: Brute-force search scenarios with high precision requirements</td><td valign="middle">● <a href="#generating-operators">INT8Flat</a><br>● <a href="#generating-operators">AICPU</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexInt8Flat.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/full_retrieval.md#ascendindexflat">AscendIndexFlat</a></td><td valign="middle">● Feature type: FP32, FP16<br>● Feature dimension: 32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096<br>● Distance type: L2 and IP<br>● Calculation precision: High<br>● Device memory usage: High<br>● Applicable scenario: Brute-force search scenarios with high precision requirements. IP distance is recommended when <code>dim &gt; 128</code>.</td><td valign="middle">● <a href="#generating-operators">Flat</a><br>● <a href="#generating-operators">AICPU</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexFlat.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/full_retrieval.md#ascendindexsq">AscendIndexSQ</a></td><td valign="middle">● Feature type: FP32<br>● Feature dimension: 64, 128, 256, 384, 512, 768<br>● Distance type: L2 and IP<br>● Calculation precision: High<br>● Device memory usage: Low, because it is quantized to int8<br>● Applicable scenario: Brute-force search scenarios with relatively high precision requirements</td><td valign="middle">● <a href="#generating-operators">SQ8</a><br>● <a href="#generating-operators">AICPU</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexSQ.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/full_retrieval.md#ascendindexcluster">AscendIndexCluster</a></td><td valign="middle">● Feature type: FP32<br>● Feature dimension: 32, 64, 128, 256, 384, 512<br>● Distance type: IP<br>● Calculation precision: High<br>● Device memory usage: Relatively high<br>● Applicable scenario: Clustering scenarios that only calculate distance<br>● Only supports Atlas Inference Series products</td><td valign="middle">● <a href="#generating-operators">Flat</a><br>● <a href="#generating-operators">AICPU</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexCluster.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/full_retrieval.md#indexil">IndexIL</a></td><td valign="middle">It needs to run on the Device. Installation and deployment are complex, so it is not recommended for now.</td><td valign="middle">- <a href="#generating-operators">Flat</a>;</td><td valign="middle" width="140" align="center">See <a href="./api/full_retrieval.md#indexilflat">IndexILFlat</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/full_retrieval.md#ascendindexilflat">AscendIndexILFlat</a></td><td valign="middle">● Feature type: FP16, FP32<br>● Feature dimension: 32, 64, 128, 256, 384, 512<br>● Distance type: IP<br>● Calculation precision: High<br>● Device memory usage: Relatively high<br>● Applicable scenario: Clustering scenarios that only calculate distance<br>● Only supports Atlas Inference Series products</td><td valign="middle">● <a href="#generating-operators">Flat</a><br>● <a href="#generating-operators">AICPU</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/tree/master/IndexSDK">Link</a></td></tr>
</tbody></table>

### Approximate Search

**Approximate Search Algorithm Introduction**

<table><tbody>
<tr><td align="center" valign="middle" width="250"><strong>Algorithm (API Reference)</strong></td><td align="center" valign="middle"><strong>Algorithm Usage Scenario</strong></td><td align="center" valign="middle"><strong>Operators to Generate</strong></td><td valign="middle" width="140" align="center"><strong>Sample Link</strong></td></tr>
<tr><td valign="middle" width="250"><a href="./api/approximate_retrieval.md#ascendindexivfsp">AscendIndexIVFSP</a></td><td valign="middle">● Feature type: FP32<br>● Feature dimension: 64, 128, 256, 512, 768<br>● Distance type: L2<br>● Calculation precision: Medium<br>● Device memory usage: Low, because features are compressed<br>● Applicable scenario: Suitable for approximate search scenarios with billion-level base libraries, high performance requirements, and tolerance for precision loss.<br>● Only supports Atlas Inference Series products</td><td valign="middle">● IVFSP service operator<br>● IVFSP AICPU operator<br>● IVFSP training operator (used only when codebook files need to be generated through training)<br>● See <a href="#generating-operators">IVFSP</a>.</td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFSP.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/approximate_retrieval.md#ascendindexivfsq">AscendIndexIVFSQ</a></td><td valign="middle">● Feature type: FP32<br>● Feature dimension: 64, 128, 256, 384, 512<br>● Distance type: L2 and IP<br>● Calculation precision: Medium<br>● Device memory usage: Low, because it is quantized to int8<br>● Applicable scenario: The IVFSQ algorithm acts as a performance-precision tradeoff, suitable for scenarios that tolerate precision loss but require high performance.</td><td valign="middle">● <a href="#generating-operators">IVFSQ8</a><br>● <a href="#generating-operators">AICPU</a><br>● <a href="#generating-operators">FlatAT</a> (generate the FlatAT operator only when <code>useKmeansPP</code> is set to <code>true</code>.)</td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFSQ.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/approximate_retrieval.md#ascendindexivfsqt">AscendIndexIVFSQT</a></td><td valign="middle">● Feature type: FP32<br>● Feature dimension: 256<br>● Distance type: IP<br>● Calculation precision: Medium<br>● Device memory usage: Low, because of quantization and dimensionality reduction<br>● Applicable scenario: AscendIndexIVFSQT is a three-stage IVFSQ retrieval algorithm that includes dimensionality reduction. It is suitable for approximate search scenarios with billion-level base libraries, high performance requirements, and tolerance for precision loss.</td><td valign="middle">● <a href="#generating-operators">IVFSQT</a><br>● <a href="#generating-operators">FlatAT</a><br>● <a href="#generating-operators">AICPU</a><br>● <a href="#generating-operators">FlatInt8AT</a> (required on Atlas Inference Series products)</td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFSQT.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/approximate_retrieval.md#ascendindexbinaryflat">AscendIndexBinaryFlat</a></td><td valign="middle">● Feature type: uint8 binary features<br>● Feature dimension: 256, 512, 1024<br>● Distance type: Hamming and IP<br>● Calculation precision: High<br>● Device memory usage: Low<br>● Applicable scenario: The AscendIndexBinaryFlat class inherits from Faiss <code>IndexBinary</code> and is used for binary feature retrieval. It suits scenarios with low memory usage requirements and high performance requirements.<br>● Only supports Atlas Inference Series products</td><td valign="middle">● <a href="#generating-operators">BinaryFlat</a><br>● <a href="#generating-operators">AICPU</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexBinaryFlat.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/approximate_retrieval.md#ascendindexvstar">AscendIndexVStar</a></td><td valign="middle">● Feature type: FP32<br>● Feature dimension: 128, 256, 512, 1024<br>● Distance type: L2<br>● Calculation precision: Medium<br>● Device memory usage: Low, because features are compressed<br>● Applicable scenario: Suitable for approximate search scenarios with tens of millions of base vectors, high performance requirements, and tolerance for precision loss.<br>● Only supports Atlas Inference Series products</td><td valign="middle">● VStar service operator<br>● VStar AICPU operator<br>● VStar training operator (used only when codebook files need to be generated through training)<br>● See <a href="#generating-operators">VSTAR</a>.</td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexVStar.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/approximate_retrieval.md#ascendindexgreat">AscendIndexGreat</a></td><td valign="middle">● Feature type: FP32<br>● Feature dimension: 128, 256, 512, 1024<br>● Distance type: L2<br>● Calculation precision: Medium<br>● Device memory usage: Low, because features are compressed<br>● Applicable scenario: Suitable for approximate search scenarios with tens of millions of base vectors, high performance requirements, and tolerance for precision loss.<br>● Only supports Atlas Inference Series products</td><td valign="middle">● VStar service operator<br>● VStar AICPU operator<br>● VStar training operator (used only when codebook files need to be generated through training)<br>● See <a href="#generating-operators">VSTAR</a>.</td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexGreat.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/approximate_retrieval.md#ascendindexivfflat">AscendIndexIVFFlat</a></td><td valign="middle">● Feature type: FP32<br>● Feature dimension: 128<br>● Distance type: IP<br>● Calculation precision: Medium<br>● Device memory usage: Medium<br>● Applicable scenario: Suitable for approximate search scenarios with billion-level base libraries, high performance requirements, and tolerance for precision loss.<br>● Only supports Atlas A2, A3, and Ascend950 inference series products</td><td valign="middle">● <a href="#generating-operators">AICPU</a><br>● <a href="#ivfflat">IVFFLAT</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFFlat.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/approximate_retrieval.md#ascendindexivfpq">AscendIndexIVFPQ</a></td><td valign="middle">● Feature type: FP32<br>● Feature dimension: 128<br>● Distance type: IP<br>● Calculation precision: Medium<br>● Device memory usage: Medium<br>● Applicable scenario: Suitable for approximate search scenarios with billion-level base libraries, high performance requirements, and tolerance for precision loss.<br>● Only supports Atlas A2, A3, and Ascend950 inference series products</td><td valign="middle">● <a href="#generating-operators">AICPU</a><br>● <a href="#ivfpq">IVFPQ</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFPQ.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/approximate_retrieval.md#ascendindexivfrabitq">AscendIndexIVFRaBitQ</a></td><td valign="middle">● Feature type: FP32<br>● Feature dimension: 128<br>● Distance type: L2 and IP<br>● Calculation precision: Medium<br>● Device memory usage: Low, because features are compressed<br>● Applicable scenario: Suitable for approximate search scenarios with billion-level base libraries, high performance requirements, and tolerance for precision loss.<br>● Only supports Atlas A2, A3, and Ascend950 inference series products</td><td valign="middle">● <a href="#generating-operators">AICPU</a><br>● <a href="#ivfrabitq">IVFRaBitQ</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFRabitQ.cpp">Link</a></td></tr>
</tbody></table>

### Attribute Filter Search

**Attribute Filter Search Algorithm Introduction**

<table><tbody>
<tr><td align="center" valign="middle" width="250"><strong>Algorithm (API Reference)</strong></td><td align="center" valign="middle"><strong>Algorithm Usage Scenario</strong></td><td align="center" valign="middle"><strong>Operators to Generate</strong></td><td valign="middle" width="140" align="center"><strong>Sample Link</strong></td></tr>
<tr><td valign="middle" width="250"><a href="./api/attribute_filtering-based_retrieval.md#ascendindexts">AscendIndexTS</a></td><td valign="middle">● Feature type: uint8 binary features, int8, FP32, depending on the algorithm<br>● Feature dimension: depends on the specific algorithm<br>● Distance type: Hamming, Cos, IP, L2<br>● Calculation precision: Relatively high<br>● Device memory usage: Relatively high<br>● Applicable scenario: Spatial-temporal library scenarios that require attribute filtering<br>● Hamming distance is supported only on Atlas Inference Series products</td><td valign="middle">● <a href="#generating-operators">Mask</a><br>● <a href="#generating-operators">BinaryFlat</a><br>● <a href="#generating-operators">Int8Flat</a><br>● <a href="#generating-operators">Flat</a><br>● <a href="#generating-operators">AICPU</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexTS.cpp">Link</a></td></tr>
</tbody></table>

### Multi-Index Batch Search

**Interface Introduction**

<table><tbody>
<tr><td align="center" valign="middle" width="250"><strong>Interface (API Reference)</strong></td><td align="center" valign="middle"><strong>Interface Usage Scenario</strong></td><td align="center" valign="middle"><strong>Algorithms That Can Use This Interface</strong></td><td valign="middle" width="140" align="center"><strong>Sample Link</strong></td></tr>
<tr><td valign="middle" width="250"><a href="./api/multi-index_batch_retrieval.md#search-faissindex">Search</a></td><td valign="middle">Search on multiple indexes on a single Device.</td><td valign="middle">● <a href="./api/full_retrieval.md#ascendindexsq">AscendIndexSQ</a><br>● <a href="./api/full_retrieval.md#ascendindexflat">AscendIndexFlat</a><br>● <a href="./api/approximate_retrieval.md#ascendindexivfsp">AscendIndexIVFSP</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/multi-index_batch_retrieval.md#search-ascendindex">Search</a></td><td valign="middle">Search on multiple AscendIndex instances on a single Device.</td><td valign="middle">● <a href="./api/full_retrieval.md#ascendindexsq">AscendIndexSQ</a><br>● <a href="./api/full_retrieval.md#ascendindexflat">AscendIndexFlat</a><br>● <a href="./api/approximate_retrieval.md#ascendindexivfsp">AscendIndexIVFSP</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/multi-index_batch_retrieval.md#search-ascendindexint8">Search</a></td><td valign="middle">Search on multiple AscendIndexInt8 instances on a single Device.</td><td valign="middle">● <a href="./api/full_retrieval.md#ascendindexint8flat">AscendIndexInt8Flat</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/multi-index_batch_retrieval.md#searchwithfilter-faissindex-single-filter">SearchWithFilter</a></td><td valign="middle">Search on multiple indexes with attribute filtering on a single Device, single filter.</td><td valign="middle">● <a href="./api/full_retrieval.md#ascendindexsq">AscendIndexSQ</a><br>● <a href="./api/approximate_retrieval.md#ascendindexivfsp">AscendIndexIVFSP</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/multi-index_batch_retrieval.md#searchwithfilter-ascendindex-single-filter">SearchWithFilter</a></td><td valign="middle">Search on multiple AscendIndex instances with attribute filtering on a single Device, single filter.</td><td valign="middle">● <a href="./api/full_retrieval.md#ascendindexsq">AscendIndexSQ</a><br>● <a href="./api/approximate_retrieval.md#ascendindexivfsp">AscendIndexIVFSP</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/multi-index_batch_retrieval.md#searchwithfilter-faissindex-multiple-filters">SearchWithFilter</a></td><td valign="middle">Search on multiple indexes with filtering attributes on a single Device, multiple filters.</td><td valign="middle">● <a href="./api/full_retrieval.md#ascendindexsq">AscendIndexSQ</a><br>● <a href="./api/approximate_retrieval.md#ascendindexivfsp">AscendIndexIVFSP</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/multi-index_batch_retrieval.md#searchwithfilter-ascendindex-multiple-filters">SearchWithFilter</a></td><td valign="middle">Search on multiple AscendIndex instances with attribute filtering on a single Device, multiple filters.</td><td valign="middle">● <a href="./api/full_retrieval.md#ascendindexsq">AscendIndexSQ</a><br>● <a href="./api/approximate_retrieval.md#ascendindexivfsp">AscendIndexIVFSP</a></td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">Link</a></td></tr>
</tbody></table>

### Other Functions

**Algorithm Introduction**

<table><tbody>
<tr><td align="center" valign="middle" width="250"><strong>Algorithm (API Reference)</strong></td><td align="center" valign="middle"><strong>Algorithm Requirements, Such as Performance and Scenario Differences</strong></td><td align="center" valign="middle"><strong>Invocation Method</strong></td><td valign="middle" width="140" align="center"><strong>Sample Link</strong></td></tr>
<tr><td valign="middle" width="250"><a href="./api/more_functions.md#ireduction">IReduction</a></td><td valign="middle">IReduction is a unified interface for dimensionality reduction methods in the feature retrieval component. It currently supports the <code>PCAR</code> and <code>NN</code> dimensionality reduction algorithms.</td><td valign="middle">Initialize it with <code>ReductionConfig</code>, call <code>CreateReduction</code> to create the reduction object, and then call <code>train</code> and <code>reduce</code>.</td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIReduction.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/more_functions.md#ascendnninference">AscendNNInference</a></td><td valign="middle">Perform inference through a neural network.</td><td valign="middle">Create the NN reduction object with <code>AscendNNInference</code>, and then call <code>infer</code> for dimensionality reduction.</td><td valign="middle" width="140" align="center"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIReduction.cpp">Link</a></td></tr>
<tr><td valign="middle" width="250"><a href="./api/more_functions.md#ascendcloner">AscendCloner</a></td><td valign="middle">Index SDK provides a way to copy retrieval Index resources on the NPU to CPU-side Faiss. The copy process happens in memory, and the data loaded in the original NPU Index is copied to CPU memory so users can perform retrieval on the CPU with the same base library.</td><td valign="middle"><code>index_ascend_to_cpu</code> copies an AscendIndex to a CPU Index. <code>index_cpu_to_ascend</code> copies a CPU Index to an AscendIndex.</td><td valign="middle" width="140" align="center">None</td></tr>
</tbody></table>

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

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 flat_generate_model.py -d &lt;dim&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;dim&gt;</code>: Feature vector dimension D. The default value is <code>512</code>.<br>● <code>&lt;core_num&gt;</code>: Number of AI Cores on the Ascend AI Processor. The default value is <code>2</code>. No additional configuration is required.<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>&lt;pool_size&gt;</code>: Process pool size for multi-process scheduling during batch operator generation. The default value is <code>10</code>.<br>● <code>&lt;npu_type&gt;</code>: Hardware form factor.<br>● For Atlas 200/300/500 Inference Products and Atlas Inference Series products, run <code>npu-smi info</code> on the server where the Ascend AI Processor is installed. Remove the last digit from the reported <code>Name</code>, and the remaining value is <code>npu_type</code>.<br>● For Atlas 800I A2 Inference Servers, run <code>npu-smi info</code> on the server where the Ascend AI Processor is installed. The reported <code>Name</code> is <code>npu_type</code>.<br>● For Atlas 800I A3 Supernode Servers, run <code>npu-smi info -t board -i 0 -c 0</code> to obtain <code>NPU Name</code>. <code>910_</code> plus the <code>NPU Name</code> value is <code>npu_type</code>.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">Running this command generates a set of distance calculation operator model files. You need to modify the parameters in the command yourself.</td></tr>
<tr><td width="140" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}. 0 ≤ <code>pool_size</code> ≤ 32</td></tr>
</tbody></table>

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

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 sq8_generate_model.py -d &lt;dim&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;dim&gt;</code>: Feature vector dimension D. The default value is <code>128</code>.<br>● <code>&lt;core_num&gt;</code>: Number of AI Cores on the Ascend AI Processor. The default value is <code>2</code>. If this parameter is not specified, it is configured according to <code>&lt;npu_type&gt;</code>: when <code>npu_type</code> is <code>310</code>, <code>&lt;core_num&gt;</code> is <code>2</code><br>● when <code>npu_type</code> is <code>310P</code>, <code>&lt;core_num&gt;</code> is <code>8</code>.<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>&lt;pool_size&gt;</code>: Process pool size for multi-process scheduling during batch operator generation. The default value is <code>10</code>.<br>● <code>&lt;npu_type&gt;</code>: Hardware form factor. Currently, <code>&lt;npu_type&gt;</code> supports Atlas 200/300/500 Inference Products and Atlas Inference Series products. The valid values are <code>310</code> and <code>310P</code>, and the default value is <code>310</code>.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">Running this command generates a set of SQ8 distance calculation operator model files. You need to modify the parameters in the command yourself.</td></tr>
<tr><td width="140" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {64, 128, 256, 384, 512, 768}. 0 ≤ <code>pool_size</code> ≤ 32</td></tr>
</tbody></table>

**Involved Algorithms**

- [AscendIndexSQ](#full-search)
- [Search (multiple-index search on a single device)](./api/multi-index_batch_retrieval.md#search-faissindex)
- [Search (multiple AscendIndex searches on a single device)](./api/multi-index_batch_retrieval.md#search-ascendindex)
- [SearchWithFilter (FaissIndex single filter)](./api/multi-index_batch_retrieval.md#searchwithfilter-faissindex-single-filter)
- [SearchWithFilter (AscendIndex single filter)](./api/multi-index_batch_retrieval.md#searchwithfilter-ascendindex-single-filter)
- [SearchWithFilter (FaissIndex multiple filters)](./api/multi-index_batch_retrieval.md#searchwithfilter-faissindex-multiple-filters)
- [SearchWithFilter (AscendIndex multiple filters)](./api/multi-index_batch_retrieval.md#searchwithfilter-ascendindex-multiple-filters)

#### IVFSQ8

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 ivfsq8_generate_model.py -d &lt;dim&gt; -c &lt;coarse_centroid_num&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;dim&gt;</code>: Feature vector dimension D. The default value is <code>128</code>.<br>● <code>&lt;coarse_centroid_num&gt;</code>: Number of L1 cluster centroids. The default value is <code>16384</code>.<br>● <code>&lt;core_num&gt;</code>: Number of AI Cores on the Ascend AI Processor. The default value is <code>2</code>. If this parameter is not specified, it is configured according to <code>&lt;npu_type&gt;</code>: when <code>npu_type</code> is <code>310</code>, <code>&lt;core_num&gt;</code> is <code>2</code><br>● when <code>npu_type</code> is <code>310P</code>, <code>&lt;core_num&gt;</code> is <code>8</code>.<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>&lt;pool_size&gt;</code>: Process pool size for multi-process scheduling during batch operator generation. The default value is <code>10</code>.<br>● <code>&lt;npu_type&gt;</code>: Hardware form factor. Currently, <code>&lt;npu_type&gt;</code> supports Atlas 200/300/500 Inference Products and Atlas Inference Series products. The valid values are <code>310</code> and <code>310P</code>, and the default value is <code>310</code>.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">Running this command generates a set of operator model files. You need to modify the parameters in the command yourself.</td></tr>
<tr><td width="140" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {64, 128, 256, 384, 512}. <code>coarse_centroid_num</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}. 0 ≤ <code>pool_size</code> ≤ 32</td></tr>
</tbody></table>

**Involved Algorithms**

[AscendIndexIVFSQ](#approximate-search)

#### INT8Flat

> [!NOTE]
>
> The main difference between INT8Flat and SQ8 is that INT8 is quantized externally, and the Index input features are of the INT8 type. SQ8 is quantized internally by the Index, and the Index input features are of the Float32 type.

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 int8flat_generate_model.py -d &lt;dim&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt; -code &lt;code_num&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;dim&gt;</code>: Feature vector dimension D. The default value is <code>512</code>.<br>● <code>&lt;core_num&gt;</code>: Number of AI Cores on the Ascend AI Processor. The default value is <code>2</code>. No additional configuration is required.<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>&lt;pool_size&gt;</code>: Process pool size for multi-process scheduling during batch operator generation. The default value is <code>10</code>.<br>● <code>&lt;npu_type&gt;</code>: Hardware form factor. For Atlas 200/300/500 Inference Products and Atlas Inference Series products, run <code>npu-smi info</code> on the server where the Ascend AI Processor is installed. Remove the last digit from the reported <code>Name</code>, and the remaining value is <code>npu_type</code>. For Atlas 800I A2 Inference Servers, run <code>npu-smi info</code> on the server where the Ascend AI Processor is installed. The reported <code>Name</code> is <code>npu_type</code>.<br>● <code>&lt;code_num&gt;</code>: The base library block size when the operator is called. The default value is <code>262144</code>. If it is not set, operators for all <code>code_num</code> values are generated by default.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">Running this command generates a set of operator model files. You need to modify the parameters in the command yourself.</td></tr>
<tr><td width="140" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {64, 128, 256, 384, 512, 768, 1024}. 0 ≤ <code>pool_size</code> ≤ 32. <code>code_num</code> ∈ {16384, 32768, 65536, 131072, 262144}</td></tr>
</tbody></table>

**Involved Algorithms**

- [AscendIndexInt8Flat](#full-search)
- [AscendIndexTS](#attribute-filter-search)
- [Search (single-device multiple AscendIndexInt8 searches)](./api/multi-index_batch_retrieval.md#search-ascendindexint8)

#### IVFSQT

> [!NOTE]
>
> To reduce the time consumed by train and add, you need to generate the FlatAT operator. The `dim` of Flat must be the same as the `dim_in` of IVFSQT, and the `code_num` of Flat must match the `coarse_centroid_num` of IVFSQT.

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 ivfsqt_generate_model.py --cores &lt;core_num&gt; -d &lt;dim_in&gt; -r &lt;compress_ratio&gt; -c &lt;coarse_centroid_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;dim_in&gt;</code>: Input feature vector dimension. The default value is <code>256</code>.<br>● <code>&lt;compress_ratio&gt;</code>: Ratio of input to output dimensions. The default value is <code>4</code>. Range: <code>compress_ratio &gt;= 1</code>.<br>● <code>&lt;coarse_centroid_num&gt;</code>: Number of L1 cluster centroids. The default value is <code>16384</code>.<br>● <code>&lt;core_num&gt;</code>: Number of AI Cores on the Ascend AI Processor. The default value is <code>2</code>. If this parameter is not specified, it is configured according to <code>&lt;npu_type&gt;</code>: when <code>npu_type</code> is <code>310</code>, <code>&lt;core_num&gt;</code> is <code>2</code><br>● when <code>npu_type</code> is <code>310P</code>, <code>&lt;core_num&gt;</code> is <code>8</code>.<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>&lt;pool_size&gt;</code>: Process pool size for multi-process scheduling during batch operator generation. The default value is <code>32</code>. Range: <code>1 &lt;= pool_size &lt;= 32</code>.<br>● <code>&lt;npu_type&gt;</code>: Hardware form factor. Currently, <code>&lt;npu_type&gt;</code> supports Atlas 200/300/500 Inference Products and Atlas Inference Series products. The valid values are <code>310</code> and <code>310P</code>, and the default value is <code>310</code>.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">Running this command generates a set of operator model files.</td></tr>
<tr><td width="140" align="center" valign="middle">Constraints</td><td valign="middle"><code>&lt;dim_in&gt;</code> ∈ {256}. <code>&lt;compress_ratio&gt;</code> ∈ {2, 4, 8}. <code>&lt;coarse_centroid_num&gt;</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}. <code>&lt;dim_in&gt;</code> can be divisible by <code>&lt;compress_ratio&gt;</code>.</td></tr>
</tbody></table>

**Involved Algorithms**

[AscendIndexIVFSQT](#approximate-search)

#### FlatAT

> [!NOTE]
>
> The current FlatAT operator is used together with IVF-type operators to speed up the `add` and `train` processes of IVF-type operators. You cannot call FlatAT directly. The current `add` and `train` acceleration feature is specified through `AscendIndexIVFConfig.useKmeansPP` in IVF. In this case, training is supported only when the training scale is less than 7,000,000.

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 flat_at_generate_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -c &lt;code_num&gt; -p &lt;process_id&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;dim&gt;</code>: Input feature vector dimension. The default value is <code>64</code>.<br>● <code>&lt;code_num&gt;</code>: Number of base library features to compare with the input feature. The default value is <code>8192</code>.<br>● <code>&lt;core_num&gt;</code>: Number of AI Cores on the Ascend AI Processor. The default value is <code>2</code>. If this parameter is not specified, it is configured according to <code>&lt;npu_type&gt;</code>: when <code>npu_type</code> is <code>310</code>, <code>&lt;core_num&gt;</code> is <code>2</code><br>● when <code>npu_type</code> is <code>310P</code>, <code>&lt;core_num&gt;</code> is <code>8</code>.<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>&lt;npu_type&gt;</code>: Hardware form factor. Currently, <code>&lt;npu_type&gt;</code> supports Atlas 200/300/500 Inference Products and Atlas Inference Series products. The valid values are <code>310</code> and <code>310P</code>, and the default value is <code>310</code>.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">Running this command generates a set of operator model files.; FlatAT operators are mainly used in IVF scenarios to reduce the time consumed by train and add.</td></tr>
<tr><td width="140" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {64, 128, 256}. <code>code_num</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}</td></tr>
</tbody></table>

**Involved Algorithms**

- [AscendIndexIVFSQ](#approximate-search)
- [AscendIndexIVFSQT](#approximate-search)

#### FlatInt8AT

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 flat_at_int8_generate_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -c &lt;code_num&gt; -p &lt;process_id&gt; --soc-version &lt;soc_version&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;core_num&gt;</code>: Number of AI Cores on the Ascend AI Processor. The default value is <code>8</code>.<br>● <code>&lt;dim&gt;</code>: Input feature vector dimension. The default value is <code>256</code>.<br>● <code>&lt;code_num&gt;</code>: Number of base library features to compare with the input feature. The default value is <code>16384</code>.<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>&lt;soc_version&gt;</code>: Model of the Ascend AI Processor. The default value is <code>Ascend310P3</code>. No additional configuration is required.<br>● <code>&lt;npu_type&gt;</code>: Hardware form factor. Currently, only Atlas Inference Series products are supported. The default value is <code>310P</code>. No additional configuration is required.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">Running this command generates a set of operator model files.; FlatInt8AT optimizes the time consumed by train, add, and update in IVFSQT for Atlas Inference Series usage scenarios.</td></tr>
<tr><td width="140" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {256}. <code>code_num</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}. <code>soc_version</code> ∈ {Ascend310P3}</td></tr>
</tbody></table>

**Involved Algorithms**

[AscendIndexIVFSQT](#approximate-search)

#### AICPU

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 aicpu_generate_model.py --cores &lt;core_num&gt; -p &lt;process_id&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;core_num&gt;</code>: Number of AI Cores on the Ascend AI Processor. The default value is <code>2</code>. (Reserved parameter, not used at present.)<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>&lt;npu_type&gt;</code>: Hardware form factor. Currently, <code>&lt;npu_type&gt;</code> supports Atlas 200/300/500 Inference Products, Atlas Inference Series products, and Atlas A2 Inference Series products. The default value is <code>310</code>. If you cannot determine the exact <code>npu_type</code>, run <code>npu-smi info</code> on the server where the Ascend AI Processor is installed. Remove the last digit from the reported <code>Name</code>, and the remaining value is <code>npu_type</code>. For Atlas 800I A3 Supernode Servers, you can run <code>npu-smi info -t board -i 0 -c 0</code> to obtain <code>NPU Name</code>. <code>910_</code> plus the <code>NPU Name</code> value is <code>npu_type</code>.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">Running this command generates a set of operator model files.; AICPU operator model files only need to be generated once, and operators for all algorithms are generated.</td></tr>
</tbody></table>

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

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 binary_flat_generate_model.py -d &lt;dim&gt; -q &lt;query_type&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;dim&gt;</code>: Dimension of the binary feature vector. <code>dim</code> ∈ {256, 512, 1024}. The default value is <code>512</code>.<br>● <code>&lt;query_type&gt;</code>: Search type. The default is <code>uint8</code>. When you improve the performance of the <code>search</code> interface of the AscendIndexBinaryFlat algorithm, set it to <code>float</code>.<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>&lt;pool_size&gt;</code>: Process pool size for multi-process scheduling during batch operator generation. The default value is <code>16</code>.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">None.</td></tr>
</tbody></table>

**Involved Algorithms**

- [AscendIndexBinaryFlat](#approximate-search)
- [AscendIndexTS](#attribute-filter-search)

#### Mask

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 mask_generate_model.py -token &lt;max_token_cnt&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;max_token_cnt&gt;</code>: Maximum number of tokens for operator generation. The default value is 2500. The recommended range is [1, 300000].<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>&lt;pool_size&gt;</code>: Process pool size for multi-process scheduling during batch operator generation. The default value is <code>16</code>.<br>● <code>&lt;npu_type&gt;</code>: Hardware form factor. Only <code>310P</code> is supported.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">None.</td></tr>
</tbody></table>

**Involved Interface**

[AscendIndexTS](#attribute-filter-search)

#### IVFSP

IVFSP search currently supports only the `310P` hardware form factor. It involves the following model file generation types:

- `ivfsp_generate_model.py`: Generates IVFSP service operator model files. For details, see [IVFSP service operator model file generation](#ivfsp-service-operator-model-file-generation).
- `ivfsp_aicpu_generate_model.py`: Generates IVFSP AICPU operator model files. For details, see [IVFSP AICPU operator model file generation](#ivfsp-aicpu-operator-model-file-generation).
- `ivfsp_generate_pyacl_model.py`: Generates the training operator model files required for IVFSP codebook training. For details, see [IVFSP training operator model file generation](#ivfsp-training-operator-model-file-generation).

##### IVFSP Service Operator Model File Generation

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 ivfsp_generate_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -nonzero_num &lt;low_dim&gt; -nlist &lt;k&gt; -handle_batch &lt;handle_batch&gt; -code_num &lt;code_num&gt; -p &lt;process_id&gt; --pool &lt;pool_size&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;core_num&gt;</code>: Number of AI Cores. The default value is <code>8</code>, and no additional configuration is required.<br>● <code>&lt;dim&gt;</code>: Feature vector dimension. The default value is <code>256</code>.<br>● <code>&lt;low_dim&gt;</code>: Number of non-zero dimensions after feature vector compression. The default value is <code>32</code>.<br>● <code>&lt;k&gt;</code>: Number of cluster centroids. Keep this consistent with <code>&lt;k&gt;</code> in IVFSP training operator model file generation. The default value is <code>1024</code>.<br>● <code>&lt;handle_batch&gt;</code>: Number of candidate buckets dispatched per calculation during retrieval. The default value is <code>32</code>.<br>● <code>&lt;code_num&gt;</code>: Maximum number of samples per bucket dispatched per calculation during retrieval. If a bucket is too large, the program automatically splits it into multiple operator dispatches for distance calculation based on <code>code_num</code>. Keep this consistent with <code>&lt;codebook_batch_size&gt;</code> in IVFSP training operator model file generation. The default value is <code>32768</code>.<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>&lt;pool_size&gt;</code>: Process pool size for multi-process scheduling during batch operator generation. The default value is <code>16</code>.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">Running this command generates a set of AI Core operator model files for IVFSP retrieval. You need to modify the command parameters yourself.</td></tr>
<tr><td width="140" align="center" valign="middle">Constraints</td><td valign="middle">When <code>dim</code> ∈ {64, 128, 256}, <code>k</code> ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}. When <code>dim</code> ∈ {512, 768}, <code>k</code> ∈ {256, 512, 1024, 2048}. <code>low_dim</code> must be a multiple of 16 and less than or equal to <code>min(128, dim)</code>. <code>handle_batch</code> must be a multiple of 16, and <code>16 &lt;= handle_batch &lt;= 240</code>. <code>0 &lt; pool_size &lt;= 32</code>.</td></tr>
</tbody></table>

##### IVFSP AICPU Operator Model File Generation

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 ivfsp_aicpu_generate_model.py --cores &lt;core_num&gt; -p &lt;process_id&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;core_num&gt;</code>: Number of AI Cores. The default value is <code>8</code>, and no additional configuration is required.<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">Running this command generates a set of IVFSP retrieval AICPU operator model files.</td></tr>
</tbody></table>

##### IVFSP Training Operator Model File Generation

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 ivfsp_generate_pyacl_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -nonzero_num &lt;low_dim&gt; -nlist &lt;k&gt; -batch_size &lt;batch_size&gt; -code_num &lt;codebook_batch_size&gt; -p &lt;process_id&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;core_num&gt;</code>: Number of AI Cores. The default value is <code>8</code>, and no additional configuration is required.<br>● <code>&lt;dim&gt;</code>: Feature vector dimension. The default value is <code>256</code>.<br>● <code>&lt;low_dim&gt;</code>: Number of non-zero dimensions after feature vector compression. The default value is <code>32</code>.<br>● <code>&lt;k&gt;</code>: Number of cluster centroids. Keep this consistent with <code>&lt;k&gt;</code> in IVFSP service operator model file generation. The default value is <code>1024</code>.<br>● <code>&lt;batch_size&gt;</code>: Batch size used during training. The default value is <code>32768</code>.<br>● <code>&lt;codebook_batch_size&gt;</code>: Maximum number of samples used to operate on the codebook each time during training. It must be a power of 2. Keep this consistent with <code>&lt;code_num&gt;</code> in IVFSP service operator model file generation. The default value is <code>32768</code>.<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">Running this command generates a set of operator model files for IVFSP retrieval. You need to modify the command parameters yourself. The generated IVFSP training operator model files are saved in the <code>op_models_pyacl</code> subdirectory of the current directory.</td></tr>
<tr><td width="140" align="center" valign="middle">Constraints</td><td valign="middle">When <code>dim</code> ∈ {64, 128, 256}, <code>k</code> ∈ {256, 512, 1024, 2048, 4096, 8192, 16384}. When <code>dim</code> ∈ {512, 768}, <code>k</code> ∈ {256, 512, 1024, 2048}. <code>low_dim</code> must be a multiple of 16 and less than or equal to <code>min(128, dim)</code>. <code>batch_size</code> must be a multiple of 16. <code>codebook_batch_size</code> must be a multiple of 16.</td></tr>
</tbody></table>

#### VSTAR

VSTAR search currently supports only Atlas Inference Series products. It involves generating the VSTAR service operator model file (`vstar_generate_models.py`). For details, see [VSTAR](#generating-operators).

The operator generation environment must match the codebook generation environment. For details, see [Overall Description](#overall-description).

##### VSTAR Service Operator Model File Generation

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 vstar_generate_models.py --dim &lt;dim&gt; --nlistL1 &lt;nlist1&gt; --subDimL1 &lt;sub_dim1&gt; --nProbeL1 &lt;nprobe1&gt; --nProbeL2 &lt;nprobe2&gt; --segmentNumL3 &lt;segment&gt; --pool &lt;pool_size&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;dim&gt;</code>: Feature vector dimension. The default value is <code>256</code>.<br>● <code>&lt;nlist1&gt;</code>: Number of first-level cluster centroids. The default value is <code>1024</code>.<br>● <code>&lt;nprobe1&gt;</code>: Number of first-level candidate buckets dispatched for each retrieval calculation. The default value is <code>[72]</code>.<br>● <code>&lt;nprobe2&gt;</code>: Number of second-level candidate buckets dispatched for each retrieval calculation. The default value is <code>[64, 296]</code>.<br>● <code>&lt;sub_dim1&gt;</code>: Dimensionality after first-level reduction during retrieval. The default value is <code>32</code>.<br>● <code>&lt;segment&gt;</code>: Number of data segments searched from <code>nprobe2</code> during retrieval. The default value is <code>[512, 1000, 1504]</code>.<br>● <code>&lt;pool_size&gt;</code>: Process pool size for multi-process scheduling during batch operator generation. The default value is <code>16</code>.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">Running this command generates a set of AI Core and AICPU operator model files for VSTAR retrieval. You need to modify the command parameters yourself.</td></tr>
<tr><td width="140" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {128, 256, 512, 1024}. <code>nlist1</code> ∈ {256, 512, 1024}. <code>sub_dim1</code> ∈ {32, 64, 128}. <code>sub_dim1</code> must be less than <code>dim</code>. <code>nprobe1</code> ∈ (16, <code>nlist1</code>]. <code>nprobe1</code> is a list of int values, and each value in the list must be a multiple of 8. <code>nprobe2</code> ∈ (16, <code>nprobe1 * n</code>]. When <code>dim</code> is 1024, <code>n</code> is 16. For other dimensions, <code>n</code> is 32. <code>nprobe2</code> is a list of int values, and each value in the list must be a multiple of 8. <code>segment</code> ∈ (100, 5000]. <code>segment</code> is a list of int values, and each value must be a multiple of 8. <code>pool_size</code> ∈ [1, 32]. Before you run the script, determine the maximum number of processes supported by the host machine and set it appropriately.</td></tr>
</tbody></table>

**Involved Algorithms**

[AscendIndexVStar](./api/approximate_retrieval.md#ascendindexvstar)

[AscendIndexGreat](./api/approximate_retrieval.md#ascendindexgreat)

#### IVFFLAT

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 ivfflat_generate_model.py -d &lt;dim&gt; -c &lt;coarse_centroid_num&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;dim&gt;</code>: Feature vector dimension. The default value is <code>128</code>.<br>● <code>&lt;coarse_centroid_num&gt;</code>: Number of first-level cluster centroids. The default value is <code>1024</code>.<br>● <code>&lt;core_num&gt;</code>: Number of AI Cores on the Ascend AI Processor. The default value is <code>40</code>. If this parameter is not specified, it is configured according to <code>&lt;npu_type&gt;</code>: when <code>&lt;npu_type&gt;</code> is <code>910B3</code>, <code>&lt;core_num&gt;</code> is <code>40</code>.<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>&lt;pool_size&gt;</code>: Process pool size for multi-process scheduling during batch operator generation. The default value is <code>10</code>.<br>● <code>&lt;npu_type&gt;</code>: Hardware form factor. Currently, <code>&lt;npu_type&gt;</code> supports Atlas A2, A3, and Ascend950 inference series products. The default value is <code>910B4</code>. If you cannot determine the specific <code>npu_type</code>, run <code>npu-smi info</code> on the server where the Ascend AI Processor is installed. Remove the last digit from the reported <code>Name</code>, and the remaining value is <code>npu_type</code>. For Atlas 800I A3 Supernode Servers, run <code>npu-smi info -t board -i 0 -c 0</code> to obtain <code>NPU Name</code>. <code>910_</code> plus the <code>NPU Name</code> value is <code>npu_type</code>. For Atlas Ascend950 Supernode Servers, set <code>npu_type</code> to <code>Ascend950PR</code>.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">Running this command generates a set of operator model files. You need to modify the parameters in the command yourself.</td></tr>
<tr><td width="140" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {64, 128, 256, 384, 512}. <code>coarse_centroid_num</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}. 0 ≤ <code>pool_size</code> ≤ 32</td></tr>
</tbody></table>

**Involved Algorithms**

[AscendIndexIVFFlat](./api/approximate_retrieval.md#ascendindexivfflat)

#### IVFPQ

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 ivfpq_generate_model.py -d &lt;dim&gt; -c &lt;nlist&gt; --cores &lt;core_num&gt; -m &lt;m&gt; -n &lt;nbit&gt; -topK &lt;topK&gt; -b &lt;blockNum&gt; -p &lt;process_id&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;dim&gt;</code>: Feature vector dimension. The default value is <code>128</code>.<br>● <code>&lt;nlist&gt;</code>: Number of first-level cluster centroids. The default value is <code>1024</code>.<br>● <code>&lt;core_num&gt;</code>: Number of AI Cores on the Ascend AI Processor. The default value is <code>40</code>. If this parameter is not specified, it is configured according to <code>&lt;npu_type&gt;</code>.<br>● <code>&lt;m&gt;</code>: Number of subspaces. The default value is <code>4</code>.<br>● <code>&lt;nbit&gt;</code>: Number of bits in the quantization centroid for each subspace. The default value is <code>8</code>, and no additional configuration is required. It also determines the number of codebook centroids, <code>ksub = 1 &lt;&lt; nbit</code>. When <code>nbit</code> is 8, <code>ksub</code> is 256.<br>● <code>&lt;topK&gt;</code>: Number of nearest candidate vectors returned for each query vector. The default value is <code>320</code>, and no additional configuration is required.<br>● <code>&lt;blockNum&gt;</code>: Number of candidate vector blocks to process. The default value is <code>128</code>, and no additional configuration is required.<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>&lt;npu_type&gt;</code>: Hardware form factor. The current default value is <code>Ascend950PR</code>. If you cannot determine the specific <code>npu_type</code>, run <code>npu-smi info</code> on the server where the Ascend AI Processor is installed. Remove the last digit from the reported <code>Name</code>, and the remaining value is <code>npu_type</code>. For Atlas 800I A3 Supernode Servers, run <code>npu-smi info -t board -i 0 -c 0</code> to obtain <code>NPU Name</code>. <code>910_</code> plus the <code>NPU Name</code> value is <code>npu_type</code>.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">Running this command generates a set of operator model files. You need to modify the parameters in the command yourself. For large <code>nlist</code> (262144, 524288) on Ascend 910B4, run <code>NPU_TYPE=910B4 bash generate_ivfpq_large_nlist_models.sh</code>.</td></tr>
<tr><td width="140" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {128}. <code>nlist</code> ∈ {1024, 2048, 4096, 8192, 16384, 262144, 524288}. <code>m</code> ∈ {2, 4, 8, 16, 32}. <code>n</code> ∈ {8}.</td></tr>
</tbody></table>

**Involved Algorithms**

[AscendIndexIVFPQ](./api/approximate_retrieval.md#ascendindexivfpq)

#### IVFRaBitQ

<table><tbody>
<tr><td width="140" align="center" valign="middle">Usage</td><td valign="middle"><strong><code>python3 ivfrabitq_generate_model.py -d &lt;dim&gt; -c &lt;coarse_centroid_num&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt; -m &lt;metric_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;dim&gt;</code>: Feature vector dimension. The default value is <code>128</code>.<br>● <code>&lt;coarse_centroid_num&gt;</code>: Number of first-level cluster centroids. The default value is <code>16384</code>.<br>● <code>&lt;core_num&gt;</code>: Number of AI Cores on the Ascend AI Processor. The default value is <code>40</code>. If this parameter is not specified, it is configured according to <code>&lt;npu_type&gt;</code>: when <code>&lt;npu_type&gt;</code> is <code>910B3</code>, <code>&lt;core_num&gt;</code> is <code>40</code>.<br>● <code>&lt;process_id&gt;</code>: Process ID for multi-process scheduling during batch operator generation. The default value is <code>0</code>. No additional configuration is required.<br>● <code>&lt;pool_size&gt;</code>: Process pool size for multi-process scheduling during batch operator generation. The default value is <code>10</code>.<br>● <code>&lt;npu_type&gt;</code>: Hardware form factor. The default value is <code>910B4</code>. If you cannot determine the specific <code>npu_type</code>, run <code>npu-smi info</code> on the server where the Ascend AI Processor is installed. Remove the last digit from the reported <code>Name</code>, and the remaining value is <code>npu_type</code>. For Atlas 800I A3 Supernode Servers, run <code>npu-smi info -t board -i 0 -c 0</code> to obtain <code>NPU Name</code>. <code>910_</code> plus the <code>NPU Name</code> value is <code>npu_type</code>.<br>● <code>&lt;metric_type&gt;</code>: Vector calculation mode. Use this to explicitly specify whether to calculate using <code>L2</code> or <code>IP</code> distance. The default value is <code>L2</code>.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="140" align="center" valign="middle">Description</td><td valign="middle">Running this command generates a set of operator model files. You need to modify the parameters in the command yourself.</td></tr>
<tr><td width="140" align="center" valign="middle">Constraints</td><td valign="middle"><code>dim</code> ∈ {128}. <code>coarse_centroid_num</code> ∈ {1024, 2048, 4096, 8192, 16384, 32768}. 0 ≤ <code>pool_size</code> ≤ 32</td></tr>
</tbody></table>

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

<table><tbody>
<tr><td width="180" align="center" valign="middle">Command reference</td><td valign="middle"><strong><code>python3 vstar_train_codebook.py --dataPath &lt;data_path&gt; --dim &lt;dim&gt; --codebookPath &lt;codebook_output_dir&gt; --nlistL1 &lt;nlist1&gt; --subDimL1 &lt;sub_dim1&gt; --device &lt;device&gt; --batchSize &lt;batch_size&gt; --sample &lt;sample&gt; --useOfflineCompile</code></strong></td></tr>
<tr><td width="180" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;data_path&gt;</code>: Path to the raw data used for codebook training. The data must exist. This parameter is required.<br>● <code>&lt;dim&gt;</code>: Feature vector dimension. Keep it consistent with the <code>&lt;dim&gt;</code> used when generating the VSTAR training operator model file. The default value is <code>256</code>.<br>● <code>&lt;codebook_output_dir&gt;</code>: Path that stores the final codebook file. Ensure that the directory exists and that the user running the program has write permission. For security hardening reasons, the directory tree cannot contain symbolic links.<br>● <code>&lt;nlist1&gt;</code>: Number of first-level cluster centroids. Keep it consistent with <code>&lt;nlist1&gt;</code> in VSTAR training operator model file generation. The default value is <code>1024</code>.<br>● <code>&lt;sub_dim1&gt;</code>: Dimensionality after first-level reduction during retrieval. Keep it consistent with <code>&lt;sub_dim1&gt;</code> in VSTAR training operator model file generation. The default value is <code>32</code>.<br>● <code>&lt;device&gt;</code>: Device logical ID. Run training on the specified Device. The default value is <code>1</code>.<br>● <code>&lt;batch_size&gt;</code>: Batch size used during training. Range: <code>(0, 10240]</code>. The default value is <code>10240</code>.<br>● <code>&lt;sample&gt;</code>: Sampling rate for the raw samples used in training. Range: <code>0 &lt; ratio &lt;= 1.0</code>. The default value is <code>1.0</code>.<br>● <code>--useOfflineCompile</code>: Controls whether to use the operator package dependency and run offline operator compilation to improve performance. Disabled by default. If you enable it, add this option to the end of the command line. For details, see the section on the <code>--useOfflineCompile</code> option in the VSTAR codebook generation overview.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="180" align="center" valign="middle">Usage instructions</td><td valign="middle">The size of the raw data in <code>&lt;data_path&gt;</code> must be less than or equal to 10 million 1024-dimensional vectors, that is, <code>10,000,000 * 1024 * 4 = 40,960,000,000</code>. Running this command generates a new <code>codebook_&lt;dim&gt;_&lt;nlist1&gt;_&lt;sub_dim1&gt;.bin</code> directory under <code>&lt;codebook_output_dir&gt;</code>. This is the codebook file required by <code>AscendIndexVStar</code> and <code>AscendIndexGreat</code>. If the codebook file already exists, it is overwritten. In that case, the user running the program should be the file owner. Before you train and generate the codebook, first refer to VSTAR and generate the training operator model files.</td></tr>
</tbody></table>

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

<table><tbody>
<tr><td width="180" align="center" valign="middle">Command reference</td><td valign="middle"><strong><code>python3 ivfsp_train_codebook.py --dim &lt;dim&gt; --nonzero_num &lt;nonzero_num&gt; --nlist &lt;nlist&gt; --num_iter &lt;num_iter&gt; --device &lt;device&gt; --batch_size &lt;batch_size&gt; --code_num &lt;code_num&gt; --ratio &lt;ratio&gt; --learn_data_path &lt;learn_data_path&gt; --codebook_output_dir &lt;codebook_output_dir&gt; --train_model_dir &lt;train_model_dir&gt;</code></strong></td></tr>
<tr><td width="180" align="center" valign="middle">Parameters</td><td valign="middle">● <code>&lt;dim&gt;</code>: Feature vector dimension. Keep it consistent with the <code>&lt;dim&gt;</code> used when generating the IVFSP training operator model file. It must be greater than 0.<br>● <code>&lt;nonzero_num&gt;</code>: Number of non-zero dimensions after feature vector compression. Keep it consistent with <code>&lt;low_dim&gt;</code> used when generating the IVFSP training operator model file. It must be greater than 0.<br>● <code>&lt;nlist&gt;</code>: Number of cluster centroids. Keep it consistent with <code>&lt;k&gt;</code> used when generating the IVFSP training operator model file. It must be greater than 0.<br>● <code>&lt;num_iter&gt;</code>: Training iteration count. The default value is 20. Setting it too large increases the training time. It must be greater than 0.<br>● <code>&lt;device&gt;</code>: Device logical ID. Run training on the specified Device. The default value is <code>0</code>.<br>● <code>&lt;batch_size&gt;</code>: Batch size used during training. Keep it consistent with <code>&lt;batch_size&gt;</code> used when generating the IVFSP training operator model file. It must be greater than 0 and less than or equal to 32768. The default value is <code>32768</code>.<br>● <code>&lt;code_num&gt;</code>: The maximum number of samples used to operate on the codebook each time. It must be a power of 2. Keep it consistent with <code>&lt;codebook_batch_size&gt;</code> used when generating the IVFSP training operator model file. It must be greater than 0 and less than or equal to 32768. The default value is <code>32768</code>.<br>● <code>&lt;ratio&gt;</code>: Sampling rate for the raw samples used in training. Range: <code>0 &lt; ratio &lt;= 1.0</code>. The default value is <code>1.0</code>.<br>● <code>&lt;learn_data_path&gt;</code>: Path to the raw feature file used for training. <code>bin</code> and <code>npy</code> formats are supported. In <code>bin</code> format, data is stored in row-major order and uses the <code>float32</code> data type.<br>● <code>&lt;codebook_output_dir&gt;</code>: Directory where the generated codebook file is output. Ensure that the directory exists and that the user running the program has write permission. For security hardening, the directory tree cannot contain symbolic links.<br>● <code>&lt;train_model_dir&gt;</code>: Directory that contains the IVFSP training operator model files.<br>● <code>--help \| -h</code>: Query help information.</td></tr>
<tr><td width="180" align="center" valign="middle">Usage instructions</td><td valign="middle">Running this command generates <code>codebook_&lt;dim&gt;_&lt;nonzero_num&gt;_&lt;nlist&gt;.bin</code> and <code>codebook_&lt;dim&gt;_&lt;nonzero_num&gt;_&lt;nlist&gt;.npy</code> in the directory corresponding to <code>&lt;codebook_output_dir&gt;</code>. The <code>codebook_&lt;dim&gt;_&lt;nonzero_num&gt;_&lt;nlist&gt;.bin</code> file is the codebook file required by <code>AscendIndexIVFSP</code>. If the codebook file already exists, it is overwritten. In that case, the user running the program should be the file owner. Before you train and generate the codebook, first generate the training operator model files by following the instructions for IVFSP training operator model file generation. The size of the data specified by <code>learn_data_path</code> must be greater than or equal to <code>nonzero_num * nlist * sizeof(float32)</code> bytes.</td></tr>
</tbody></table>

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
