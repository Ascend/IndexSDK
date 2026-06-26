# Appendix<a name="ZH-CN_TOPIC_0000001698168789"></a>

## Public Network Addresses Included in the Software<a name="ZH-CN_TOPIC_0000001664167368"></a>

For the public network addresses included in the software, see [Index_SDK_Public_Network_Addresses.xlsx](./resource/Index_SDK_Public_Network_Addresses.xlsx).

## Environment Variables<a name="ZH-CN_TOPIC_0000002062005944"></a>

The program uses the following environment variables when it reads them. Do not modify them.

**Table 1**  Environment variables

|Environment Variable|Description|
|--|--|
|PATH|Path to the executable.|
|LD_LIBRARY_PATH|Path to dynamic libraries.|
|PYTHONPATH|Default search path for Python module files.|
|HOME|Current user's home directory.|
|PWD|Current working directory.|
|TMPDIR|Temporary directory.|
|LANG|Locale.|

## Code Reference<a name="ZH-CN_TOPIC_0000001456375372"></a>

### Introduction<a name="ZH-CN_TOPIC_0000001456375408"></a>

The manual provides customers with the best-performing Index algorithms for exact and approximate scenarios. Users can refer to the sample code in this chapter and the open-source Faiss code to develop their own applications. The same applies to the other algorithms.

> [!NOTE]
>
>- Note that AscendFaiss/Faiss must run within `try`/`catch` blocks, and you must invoke them and handle exceptions in the manner recommended in the examples.
>- Ensure that the operator file for the corresponding dimension has been generated and deployed before you run the following code.

### AscendIndexSQ<a name="ZH-CN_TOPIC_0000001456694884"></a>

The AscendIndexSQ small-library algorithm can train on a data set and generate an appropriate quantization function. For input float32 feature vectors, AscendIndexSQ quantizes them into Int8 feature vectors and stores them on the device side to further compress storage space. When performing vector comparison, it dequantizes the Int8 vectors back to the original feature vectors for subsequent computation. A typical AscendIndexSQ sample is as follows.

```cpp
#include <faiss/ascend/AscendIndexSQ.h>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    const size_t dim = 512;
    const size_t ntotal = 10000;
    vector<float> data(dim * ntotal);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = drand48();
    }

    const size_t k = 100;
    const size_t searchNum  = 100;
    vector<float> dist(k * searchNum);
    vector<long> indices(k * searchNum);

    cout << "Search data set successfully." << endl;

    faiss::ascend::AscendIndexSQ *index = nullptr;
    try {
        faiss::ascend::AscendIndexSQConfig chipConf{0};
        index = new faiss::ascend::AscendIndexSQ(dim, faiss::ScalarQuantizer::QuantizerType::QT_8bit, faiss::METRIC_L2, chipConf);
        index->train(ntotal, data.data());
        index->add(ntotal, data.data());
        index->search(searchNum, data.data(), k, dist.data(), indices.data());
    } catch (...) {
        cout << "Exception caught!" << endl;
        delete index;
        return -1;
    }
    delete index;
    cout << "Search finished successfully" << endl;
    return 0;
}
```

### IndexILFlat<a name="ZH-CN_TOPIC_0000001506334833"></a>

IndexILFlat is a retrieval solution that runs entirely on the device side. It uses resources such as Ascend AI Processors and AI Cores to support multiple interfaces. The program needs to be compiled on the host side to generate a binary file, which is then deployed to the device side together with the related runtime dependencies. For deployment, see the usage guide below. For interface usage constraints, see [IndexILFlat](./api/full_retrieval.md#indexilflat-api).

**Reference Sample Code<a id="section15454820982"></a>**

```cpp
#include <IndexILFlat.h>
#include <iostream>
#include <numeric>
#include <vector>
#include "acl/acl.h"
#include "arm_fp16.h"
int TestComputeDistance(ascend::IndexILFlat &index, int queryN, int baseSize, float16_t *queryData)
{
    int baseSizeAlign = (baseSize + 15) / 16 * 16;
    std::vector<float> distances(queryN * baseSizeAlign);
    auto ret = index.ComputeDistance(queryN, queryData, distances.data());
    return ret;
}
int TestSearchByThreshold(ascend::IndexILFlat &index, int queryN, float16_t *queryData)
{
    int topK = 10;
    float threshold = 0.6;
    std::vector<int> num(queryN);
    std::vector<float> distances(queryN * topK);
    std::vector<ascend::idx_t> idxs(queryN * topK);
    auto ret = index.SearchByThreshold(queryN, queryData, threshold, topK, num.data(), idxs.data(), distances.data());
    return ret;
}
int main(int argc, char **argv)
{
    // 0.1 Remember to set device first, please refer to CANN Application
    // Software Development Guide (C&C++, Inference)
    aclError aclSet = aclrtSetDevice(0);
    if (aclSet != 0) {
        printf("Set device failed ,error code:%d\n", aclSet);
        return 0;
    }
    // 0.2 construct index
    const int dim = 512;
    const int baseSize = 100000;
    const int queryN = 64;
    const int capacity = 100000;
    const int resourceSize = -1;
    auto metricType = ascend::AscendMetricType::ASCEND_METRIC_INNER_PRODUCT;
    std::vector<float16_t> base(baseSize * dim);
    std::vector<ascend::idx_t> ids(baseSize);
    for (size_t j = 0; j < base.size(); j++) {
        base[j] = drand48();
    }
    std::iota(ids.begin(), ids.end(), 0);
    // 1. build Index and initialize
    ascend::IndexILFlat indexFlat;
    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret != 0) {
        printf("Index initialize failed ,error code:%d\n", ret);
        aclrtResetDevice(0);
        return 0;
    }
    // 2. add base vectors
    ret = indexFlat.AddFeatures(baseSize, base.data(), ids.data());
    if (ret != 0) {
        printf("Add features failed ,error code:%d\n", ret);
        indexFlat.Finalize();
        aclrtResetDevice(0);
        return 0;
    }
    // 3.1 Test ComputeDistance
    std::vector<float16_t> queries(queryN * dim);
    for (size_t i = 0; i < queries.size(); i++) {
        queries[i] = drand48();
    }
    ret = TestComputeDistance(indexFlat, queryN, baseSize, queries.data());
    if (ret != 0) {
        printf("Compute distance failed ,error code:%d\n", ret);
        indexFlat.Finalize();
        aclrtResetDevice(0);
        return 0;
    }
    // 3.2 Test SearchByThreshold
    ret = TestSearchByThreshold(indexFlat, queryN, queries.data());
    if (ret != 0) {
        printf("Search by threshold failed ,error code:%d\n", ret);
        indexFlat.Finalize();
        aclrtResetDevice(0);
        return 0;
    }
    // 4. release resource
    indexFlat.Finalize();
    aclrtResetDevice(0);
    printf("------------Demo correct--------------\n");
    return 0;
}
```

**Usage Guide<a name="section17204881493"></a>**

1. The IndexILFlat release package is available in the installation directory after the Index SDK software package is installed.
    - Header file: `mxIndex/device/include/IndexILFlat.h`
    - Dynamic library: `mxIndex/device/lib/libascendfaiss_minios.so`

2. The code must be compiled with the HCC compiler built into CANN. By default, the compiler path under the CANN installation directory is "/usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++". After compilation, deploy it to the device side. For details, see the "Customizing the File System" section in the *CANN Software Installation Guide (Open State, Atlas Inference Series Product)*.

    If you need to copy dependencies directly to the device side through the SSH service or log in to the Device over SSH to run the sample directly, you must refer to the "Using DSMI Interface to Enable SSH Service" section in the *CANN Software Installation Guide (Open State, Atlas Inference Series Product)* to remove the 50 MB memory limit of the SSH service. Otherwise, you cannot send all dependent files, and the use case cannot be executed.

3. Generate Operator OM Files.

    Run the following command. The relevant operator files will be generated in the `mxIndex/modelpath` directory on the host side, and the operators must also be deployed to the device side.

    ```bash
    cd mxIndex
    ./ops/custom*
    cd tools
    python3.7 flat_generate_model.py -d 512 --cores 8 -pool 16 -t 310P
    mv op_models/*.om ../modelpath
    ```

4. Compile the use case code.

    In the Index SDK project, create the `mxIndex/test` test directory. In that directory, create the `IndexILDemo.cpp` source file, copy the [reference sample code](#section15454820982), and refer to the following compilation command.

    ```bash
    /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ -fPIC -fPIE -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2 \
    -o IndexILDemo IndexILDemo.cpp \
    -fopenmp -O3 -frename-registers -fpeel-loops -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -pie -s \
    -I/usr/local/AscendMiniOs/acllib/include/ \
    -I../include \
    -I../device/include \
    -L../device/lib \
    -L/usr/local/AscendMiniOs/acllib/lib64/stub \
    -L/usr/local/Ascend/driver/lib64/common \
    -lascendcl -lascend_hal -lc_sec -lascendfaiss_minios
    ```

5. Deploy dependencies.
    - Configure `modelpath` at the same level as the executable binary file to place the operator files generated in Step 3.
    - For CANN-related library dependencies, deploy `/usr/local/AscendMiniOs/aarch64-linux/lib64` to the device side and configure it in `LD_LIBRARY_PATH`.
    - Deploy the dynamic library (`mxIndex/device/lib/libascendfaiss_minios.so`) to the device side and configure it in the `LD_LIBRARY_PATH` environment variable.

## Running Retrieval Services on the Device Side<a name="ZH-CN_TOPIC_0000001696207262"></a>

Currently, retrieval supports only the standard mode, that is, running retrieval services on the host side. However, some application scenarios require running retrieval services on the device side. The following describes how to perform retrieval on the device side.

**Prerequisites<a name="section178968232301"></a>**

- CANN has been installed following the Open State process, and the "/usr/local/AscendMiniOSRun/" folder already exists. For details, see the *CANN Software Installation Guide (Open State, Atlas Inference Series Product)*.
- The 50M memory limit of the SSH service has been lifted to ensure that all dependent files can be sent. For details, see the "Using the DSMI Interface to Enable the SSH Service" section in the *CANN Software Installation Guide (Open State, Atlas Inference Series Product)*.
- The host side must use the Arm architecture.
- P2P memory reserves 4 GB on the device side, and this portion of memory is unavailable by default. To use this memory and achieve the maximum library capacity, run the `npu-smi info set -t p2p-mem-cfg -i "id" -d "value"` command to set the chip BAR space copy enable status to "disabled". For details about command usage, see the "[Setting the Chip BAR Space Copy Enable Status for a Specified Chip](https://support.huawei.com/enterprise/zh/doc/EDOC1100523602/dbbc4954)" section in the *Atlas Center Inference Card 25.3.RC1 npu-smi Command Reference*.

**Procedure<a name="section16775174716308"></a>**

1. Generate the operators required by the algorithm to be run. For details about the algorithm, see [Algorithm Introduction](./user_guide.md#algorithm-introduction).
2. Transfer the following dependent libraries to the device side.
    - openblas: `/opt/OpenBLAS/lib`
    - Faiss: `/usr/local/faiss/faiss1.10.0/lib`
    - Runtime toolkit shared objects: `/usr/local/AscendMiniOSRun/acllib/lib64` and `/usr/local/AscendMiniOSRun/aarch64-linux/data`
    - Retrieval shared objects: `${MX_INDEX_HOME}/mxIndex/host/lib`, where `${MX_INDEX_HOME}` is the installation directory of Index SDK.
    - libgfortran.so in the Host-side compiler: `/usr/lib/aarch64-linux-gnu/libgfortran.so*`
    - Binary compiled from the demo
    - The `latest/opp/version.info` file in the toolkit directory.
    - Operator File: `${MX_INDEX_HOME}/modelpath/`

        > **NOTE**
        > The Operator File must contain only operators for the Atlas Inference Series products. Operators for other products are not allowed, because they may cause the Runtime Check on the device side to fail.

3. Log in to the device side and configure the following environment variables.

    ```bash
    # Configuring Environment Variables
    export LD_LIBRARY_PATH=./lib:./lib64:./
    # Configuring the Directory Where the version.info File Is Located
    export ASCEND_OPP_PATH=./
    ```

4. Log in to the device side and run the use case.

## Revision History<a name="ZH-CN_TOPIC_0000001682175202"></a>

|Release Date|Revision History|
|--|--|
|2025-12-30|First official release.|
