# Quick Start

This tutorial uses the [Flat](./api/01_full_retrieval/08_AscendIndexFlat.md#ascendindexflat) index type as an example to help you quickly learn how to use Index SDK in a Docker container.

## Prerequisites

Before you start, ensure that the following requirements are met:

- **Hardware**: Atlas inference products, Atlas 800I A2 inference servers, and Atlas 800I A3 SuperPoD servers are supported. For details, see [Supported Hardware and Operating Systems](./01_introduction.md#supported-hardware-and-operating-systems).
- **Docker**: Docker is installed and properly configured, and the current user can run containers.
- **Ascend Driver**: The Ascend driver is installed and properly configured. You can run the `npu-smi info` command to query NPU information.

## Step 1: Pulling the Image

### 1.1 Selecting the Matching Version

   - Select the corresponding image version based on the hardware model. Pay attention to the Ascend AI Processor model.
   - Select the corresponding image pull command based on the Ascend AI Processor model:

    For Atlas inference products, run the `npu-smi info` command on a server where the Ascend AI Processor is installed. Remove the last digit from the queried `Name` value. The result is the value of `npu_type`.
    For Atlas 800I A2 inference servers, run the `npu-smi info` command on a server where the Ascend AI Processor is installed. The queried `Name` value is the value of `npu_type`.
    For Atlas 800I A3 SuperPoD servers, run the `npu-smi info -t board -i 0 -c 0` command to query the NPU Name. The `910_NPU Name` value is the value of `npu_type`.

   | Chip Model | Pull Command |
   | ---------- | ------------ |
   | 310P | `docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/indexsdk:26.1.0-cann9.1.0-310p-ubuntu22.04-py3.12` |
   | A3 | `docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/indexsdk:26.1.0-cann9.1.0-a3-ubuntu22.04-py3.12` |
   | 910B | `docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/indexsdk:26.1.0-cann9.1.0-910b-ubuntu22.04-py3.12` |

### 1.2 Performing Environment Prechecks

    - Run the `npu-smi info` command to verify the NPU driver status.
    - Check whether the driver version is compatible with the CANN version in the image. For details, see the [Firmware and Drivers](https://www.hiascend.com/hardware/firmware-drivers) documentation. If the driver version installed in the current environment is not compatible, update the NPU driver to the corresponding version. For instructions on updating the NPU driver, see the [Driver and Firmware Installation Guide](https://support.huawei.com/enterprise/zh/doc/EDOC1100568434/36e8d875?idPath=23710424|251366513|254884019|261408772|252764743).

## Step 2: Running the Container

### 2.1 Manually Mounting Devices

- **Device mounting**: Use the `--device` parameter to map device files on the host to the container so that the container can access the specified hardware resources. `/dev/davinci` is an NPU accelerator device and can be mounted as needed. `/dev/davinci_manager`, `/dev/devmm_svm`, and `/dev/hisi_hdc` are NPU management devices and must all be mounted.

- **Driver and toolchain mounting**: Mount the driver files and toolchain directories on the host, such as `/usr/local/Ascend/driver` and `/usr/local/bin/npu-smi`, to the container in read-only mode to ensure that the runtime environment in the container is consistent with that on the host. In the following example, `/dev/davinci0` indicates that device 0 is mounted.

- **Description**: Append the corresponding image tag to `-it`. For example: `-it swr.cn-south-1.myhuaweicloud.com/ascendhub/indexsdk:26.1.0-cann9.1.0-910b-ubuntu22.04-py3.12`

```bash
docker run \
    --name index_container \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it swr.cn-south-1.myhuaweicloud.com/ascendhub/indexsdk:26.1.0-cann9.1.0-910b-ubuntu22.04-py3.12 bash

```

### 2.2 Checking Whether the Driver Is Properly Mounted in the Container

```bash
npu-smi info # NPU information is displayed without errors.
```

## Step 3: Generating Operators

- Generate a 512-dimensional Flat operator:

```bash
cd /usr/local/Ascend/mxIndex/ops && ./custom_opp_*.run
cd /usr/local/Ascend/mxIndex/tools
python3 aicpu_generate_model.py -t <npu_type>
python3 flat_generate_model.py -d 512 -t <npu_type>

# Example: 910B4 on an Atlas 800I A2 inference server:
# python3 aicpu_generate_model.py -t 910B4
# python3 flat_generate_model.py -d 512 -t 910B4

# MX_INDEX_MODELPATH is the path for storing operators. In this example, /home/Ascend/modelpath is used.
export MX_INDEX_MODELPATH=/home/Ascend/modelpath
mkdir -p ${MX_INDEX_MODELPATH}
# Move operator models to the MX_INDEX_MODELPATH directory.
mv op_models/* $MX_INDEX_MODELPATH
```

## Step 4: Testing the Example

1. Use the Flat algorithm to run an example test with a base database of 1 million entries, a feature dimension of 512, 128 search vectors, and a TopK of 10. Create a `demo.cpp` file with the following content:

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
    // Generate base database data using random numbers.
    void Generate(size_t ntotal, std::vector<float> &data, int seed = 5678)
    {
        std::default_random_engine e(seed);
        std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
        data.resize(ntotal);
        for (size_t i = 0; i < ntotal; ++i) {
            data[i] = static_cast<float>(255 * rCode(e) - 128);
        }
    }
    // Normalize the base database data.
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
            // Add the base database.
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

2. Compile `demo.cpp`:

    ```bash
    export MX_INDEX_INSTALL_PATH=/usr/local/Ascend/mxIndex
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann

    g++ --std=c++11 -fPIC -fPIE -fstack-protector-all -Wall -D_FORTIFY_SOURCE=2 -O3 -Wl,-z,relro,-z,now,-z,noexecstack -s -pie \
    -o demo demo.cpp \
    -I$MX_INDEX_INSTALL_PATH/include \
    -I/usr/local/faiss/include \
    -I/usr/local/Ascend/driver/include \
    -I/opt/OpenBLAS/include \
    -L$MX_INDEX_INSTALL_PATH/host/lib \
    -L/usr/local/faiss/lib \
    -L/usr/local/Ascend/driver/lib64 \
    -L/usr/local/Ascend/driver/lib64/driver \
    -L/opt/OpenBLAS/lib \
    -L$ASCEND_HOME_PATH/lib64 \
    -lfaiss -lascendfaiss -lopenblas -lc_sec -lascendcl -lascend_hal -lascendsearch -lock_hmm
    ```

3. Run `demo`:

    ```bash
    ./demo

    # The following information indicates that the program runs successfully.
    ...
    search end!
    ```
