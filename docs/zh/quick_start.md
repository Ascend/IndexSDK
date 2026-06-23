# 快速入门

本教程以 Flat 索引类型为例，帮助您通过 Docker 容器快速掌握 Index SDK 的基本使用方法，如需在宿主机原生安装 Index SDK，请参考[安装指南](./installation_guide.md)。

## 前置条件

开始之前请确认：

- **硬件**：支持 Atlas 推理系列产品、Atlas 800I A2、Atlas 800I A3
- **Docker**：已安装并正确配置 Docker 环境，且当前用户可运行容器。

## 步骤 1：拉取镜像

1. **选择匹配版本**
   - 访问 [Index SDK 昇腾社区镜像仓](https://www.hiascend.com/developer/ascendhub/detail/7f91c3663b5d4a97b3ae40e3cabbb3a2)
   - 根据当前硬件型号（如 Atlas 800I A2 推理服务器）选择对应的镜像版本。
   - 注意区分 CPU 架构（x86_64/aarch64）和昇腾芯片型号（Ascend 310P/910 等）

2. **环境预检查**
   - 使用 `npu-smi info` 命令验证 NPU 驱动状态
   - 检查驱动版本与镜像 CANN 版本匹配性（参考《[固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community)》文档，若未安装 NPU 驱动和固件，请先[安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)）

3. **镜像拉取示例**

   ```bash
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/indexsdk:26.0.0-910b-ubuntu22.04-py3.11
   ```

## 步骤 2：运行容器

参考 [Index SDK 昇腾社区镜像仓](https://www.hiascend.com/developer/ascendhub/detail/7f91c3663b5d4a97b3ae40e3cabbb3a2)中“运行 Index 容器”章节，正确挂载 NPU 设备和驱动，创建并运行容器。

## 步骤 3：算子生成

- 以 910B4 为例（若为其他 NPU 型号，请参考[此文档](https://gitcode.com/Ascend/IndexSDK/blob/master/docs/zh/user_guide.md#flat)修改），生成 512 维 flat 算子：

```bash
cd /usr/local/Ascend/mxIndex/ops && ./custom_opp_*.run
cd /usr/local/Ascend/mxIndex/tools
python3 aicpu_generate_model.py -t 910B4
python3 flat_generate_model.py -d 512 -t 910B4

# 将算子模型移动到MX_INDEX_MODELPATH目录
mv op_models/* $MX_INDEX_MODELPATH
```

## 步骤 4：用例测试

1. 使用 Flat（暴力检索）算法进行示例测试：底库大小 100 万条，特征维度 512 维，检索向量数 128 个，TopK 为 10。创建 demo.cpp 文件，内容如下：

    ```cpp
    #include <faiss/ascend/AscendIndexFlat.h>
    #include <sys/time.h>
    #include <random>
    // 获取当前时间
    inline double GetMillisecs()
    {
        struct timeval tv = {0, 0};
        gettimeofday(&tv, nullptr);
        return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
    }
    // 使用随机数构造底库数据
    void Generate(size_t ntotal, std::vector<float> &data, int seed = 5678)
    {
        std::default_random_engine e(seed);
        std::uniform_real_distribution<float> rCode(0.0f, 1.0f);
        data.resize(ntotal);
        for (size_t i = 0; i < ntotal; ++i) {
            data[i] = static_cast<float>(255 * rCode(e) - 128);
        }
    }
    // 底库数据归一化
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
            // index初始化
            faiss::ascend::AscendIndexFlatConfig conf(device, resourceSize);
            auto metricType = faiss::METRIC_INNER_PRODUCT;
            faiss::ascend::AscendIndexFlat index(dim, metricType, conf);
            index.reset();
            // add底库
            printf("add start!\r\n");
            index.add(ntotal, features.data());
            size_t tmpTotal = index.getBaseSize(0);
            if (tmpTotal != ntotal) {
                printf("------- Error -----------------\n");
                return -1;
            }
            // search
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

2. 编译 demo.cpp：

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

3. 运行 demo：

    ```bash
    ./demo

    # 出现以下信息，说明运行成功
    ...
    search end!
    ```
