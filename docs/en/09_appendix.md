# Appendix<a name="en-us_TOPIC_0000001698168789"></a>

## Public Network Addresses Included in the Software<a name="en-us_TOPIC_0000001664167368"></a>

For the public network addresses included in the software, see [Index_SDK_Public_Network_Addresses.xlsx](./resource/Index_SDK_Public_Network_Addresses.xlsx).

## Environment Variables<a name="en-us_TOPIC_0000002062005944"></a>

The following environment variables are used by the program.

**Table 1** Environment variables

<table><tbody>
<tr><td width="200" align="center" valign="middle"><strong>Environment Variable</strong></td><td align="center" valign="middle"><strong>Description</strong></td></tr>
<tr><td width="200" valign="middle">PATH</td><td valign="middle">Path to the executable.</td></tr>
<tr><td width="200" valign="middle">LD_LIBRARY_PATH</td><td valign="middle">Path to dynamic libraries.</td></tr>
<tr><td width="200" valign="middle">PYTHONPATH</td><td valign="middle">Default search path for Python module files.</td></tr>
<tr><td width="200" valign="middle">HOME</td><td valign="middle">Current user&#x27;s home directory.</td></tr>
<tr><td width="200" valign="middle">PWD</td><td valign="middle">Current working directory.</td></tr>
<tr><td width="200" valign="middle">TMPDIR</td><td valign="middle">Temporary directory.</td></tr>
<tr><td width="200" valign="middle">LANG</td><td valign="middle">Locale.</td></tr>
</tbody></table>

### IVFRaBitQ Debugging Environment Variables<a name="ivfrabitq-debug-env"></a>

The following environment variables are used for `AscendIndexIVFRaBitQ` development and debugging. They are **not set by default** (disabled). For details about their usage and log interpretation, see [IVFRaBitQ Runtime Diagnostics](./08_common_operations.md#ivfrabitq-runtime-debug).

**Table 2** IVFRaBitQ debugging environment variables

|Environment Variable|Description|
|--|--|
|IVFRABITQ_VERIFY_COARSE_CENTER|When set to a non-`0` value, performs D2H sampling or full verification at each stage of coarse centroid upload (H2D, rotate, and LUT).|
|IVFRABITQ_DEBUG_L1_PROBE|`1` prints the probe list; `stats` prints the tile distribution; `full` includes golden comparison and complete statistics.|
|IVFRABITQ_VERIFY_L1_DIST|When set to a non-`0` value, compares the CPU golden results with the NPU distance/probe results during the L1 search stage.|

## Code Reference<a name="en-us_TOPIC_0000001456375372"></a>

### Introduction<a name="en-us_TOPIC_0000001456375408"></a>

The manual provides the best-performing Index algorithms recommended for brute-force search and approximate nearest neighbor search scenarios. Users can refer to the code sample in this chapter and the open-source Faiss code to develop their own applications. The same applies to other algorithms.

> [!NOTE]
>
> - Note that **AscendFaiss**/**Faiss** must run within **`try`**/**`catch`** blocks, and they must be invoked and exceptions handled as recommended in the examples.
> - Ensure that the operators for the corresponding dimensions have been generated and deployed before running the following code.

### `AscendIndexSQ`<a name="en-us_TOPIC_0000001456694884"></a>

The `AscendIndexSQ` small-library algorithm can be trained on a set of data to generate an appropriate quantization function. For input float32 feature vectors, `AscendIndexSQ` quantizes them into Int8 feature vectors and stores them on the device side to further reduce storage requirements. When performing vector comparison, it dequantizes the Int8 vectors back to the original feature vectors for subsequent computation. A typical `AscendIndexSQ` sample is as follows.

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
````

### `IndexILFlat`<a name="en-us_TOPIC_0000001506334833"></a>

`IndexILFlat` is a retrieval solution that runs entirely on the device side. It uses resources such as Ascend AI Processors and AI Cores to enable its various interfaces. The program must be compiled on the host side to generate a binary file, which is then deployed to the device side together with the required runtime dependencies. For deployment, see the usage instructions below. For interface usage constraints, see [`IndexILFlat`](./api/01_full_retrieval/14_IndexILFlat.md#indexilflat).

**Reference Code Sample<a id="section15454820982"></a>**

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
    // 0.2 Construct index
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
    // 1. Build Index and initialize
    ascend::IndexILFlat indexFlat;
    auto ret = indexFlat.Init(dim, capacity, metricType, resourceSize);
    if (ret != 0) {
        printf("Index initialize failed ,error code:%d\n", ret);
        aclrtResetDevice(0);
        return 0;
    }
    // 2. Add base vectors
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

**Usage Instructions<a name="section17204881493"></a>**

1. The `IndexILFlat` release package is available in the installation directory after the Index SDK software package is installed.
    - Header file: `mxIndex/device/include/IndexILFlat.h`
    - Dynamic library: `mxIndex/device/lib/libascendfaiss_minios.so`

2. The code must be compiled using the HCC compiler built into CANN. By default, the compiler path under the CANN installation directory is `/usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++`. After compilation, deploy the executable to the device side. For details, see the [Customizing a File System](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/openform/instguide310/instgopen_0007.html) section in *CANN Software Installation (Open Form, Atlas Inference Products)*.

    To copy dependencies directly to the device side through the SSH service or log in to the device over SSH to run the sample directly, refer to the [Enabling the SSH Service Using the DSMI API](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/openform/instguide310/instgopen_0056.html) section in *CANN Software Installation (Open Form, Atlas Inference Products)* to remove the 50 MB memory limit of the SSH service. Otherwise, you cannot send all dependency files, and the sample cannot be executed.

3. Generate the operator OM files.

    Run the following commands to generate the required operator files in the `mxIndex/modelpath` directory on the host side. The operators must also be deployed to the device side.

    ```bash
    cd mxIndex
    ./ops/custom*
    cd tools
    python3.7 flat_generate_model.py -d 512 --cores 8 -pool 16 -t 310P
    mv op_models/*.om ../modelpath
    ```

4. Compile the code sample.

   In the Index SDK project, create the `mxIndex/test` directory. In this directory, create the `IndexILDemo.cpp` source file, copy the [reference code sample](#section15454820982), and compile it using a command similar to the following.

    ```bash
    /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ -fPIC -fPIE -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2 \
    -o IndexILDemo IndexILDemo.cpp \
    -fopenmp -O3 -frename-registers -fpeel-loops -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -pie -s \
    -I/usr/local/AscendMiniOSRun/acllib/include/ \
    -I../include \
    -I../device/include \
    -L../device/lib \
    -L/usr/local/AscendMiniOSRun/acllib/lib64/stub \
    -L/usr/local/Ascend/driver/lib64/common \
    -lascendcl -lascend_hal -lc_sec -lascendfaiss_minios
    ```

5. Deploy the dependencies.
    - Configure the `modelpath` directory at the same level as the executable binary and place the operator files generated in Step 3 in this directory.
    - For CANN-related library dependencies, deploy `/usr/local/AscendMiniOSRun/aarch64-linux/lib64` to the device side and add it to `LD_LIBRARY_PATH`.
    - Deploy the dynamic library `mxIndex/device/lib/libascendfaiss_minios.so` to the device side and add it to `LD_LIBRARY_PATH`.

## Running Retrieval Services on the Device Side<a name="en-us_TOPIC_0000001696207262"></a>

Currently, retrieval supports only the standard mode, in which retrieval services run on the host side. However, some application scenarios require retrieval services to run on the device side. This section describes how to run retrieval services on the device side.

**Prerequisites<a name="section178968232301"></a>**

- CANN has been installed following the Open Form procedure, and the `/usr/local/AscendMiniOSRun/` directory already exists. For details, see *[CANN Software Installation (Open Form, Atlas Inference Products)](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/openform/instguide310/instgopen_0002.html)*.
- The 50 MB memory limit of the SSH service has been removed to ensure that all dependency files can be sent. For details, see the [Enabling the SSH Service Using the DSMI API](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/openform/instguide310/instgopen_0056.html) section in *CANN Software Installation (Open Form, Atlas Inference Products)*.
- The host side must use the Arm architecture.
- 4 GB of P2P memory is reserved on the device side, and this memory is unavailable by default. To use this memory and achieve the maximum library capacity, run the `npu-smi info set -t p2p-mem-cfg -i "id" -d "value"` command to set the chip BAR space copy enable status to `disabled`. For details about the command, see the [Enabling or Disabling the Copy into the BAR Space of a Chip](https://support.huawei.com/enterprise/zh/doc/EDOC1100523602/dbbc4954) section in *Atlas Center Inference Card 25.3.RC1 npu-smi Command Reference*.

**Procedure<a name="section16775174716308"></a>**

1. Generate the operators required for the algorithm. For details about the algorithm, see [Algorithm Introduction](./05_user_guide.md#algorithm-introduction).
2. Transfer the following dependencies to the device side.
    - openblas: `/opt/OpenBLAS/lib`
    - Faiss: `/usr/local/faiss/faiss1.10.0/lib`
    - Runtime toolkit shared objects: `/usr/local/AscendMiniOSRun/acllib/lib64` and `/usr/local/AscendMiniOSRun/aarch64-linux/data`
    - Retrieval shared objects: `${MX_INDEX_HOME}/mxIndex/host/lib`, where `${MX_INDEX_HOME}` is the installation directory of Index SDK.
    - `libgfortran.so` in the host-side compiler: `/usr/lib/aarch64-linux-gnu/libgfortran.so*`
    - Demo binary
    - `latest/opp/version.info` under the toolkit directory
    - Operator files: `${MX_INDEX_HOME}/modelpath/`

        > [!NOTE]
        > The operator files must contain only operators for the Atlas inference products. Operators for other products are not allowed, as they may cause the device-side execution to fail.

3. Log in to the device side and configure the following environment variables.

    ```bash
    # Configure environment variables
    export LD_LIBRARY_PATH=./lib:./lib64:./
    # Configure the directory containing the version.info file
    export ASCEND_OPP_PATH=./
    ```

4. Log in to the device side and run the sample using the following command: `./IndexILDemo`

## Revision History<a name="en-us_TOPIC_0000001682175202"></a>

| Release Date | Revision History        |
| ------------ | ----------------------- |
| 2025-12-30   | First official release. |
