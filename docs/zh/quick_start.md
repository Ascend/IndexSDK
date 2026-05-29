# 快速入门

本教程主要包含三个部分，以Flat索引类型为例，帮助您快速掌握IndexSDK的基本使用方法：

1. 环境准备：拉取IndexSDK镜像，检测NPU环境。
2. 算子生成：生成Flat索引类型的算子。
3. 用例测试：使用生成的算子进行检索测试。

## 环境准备

- 请正确安装NPU驱动和固件，具体参见[商用版](https://www.hiascend.com/document/detail/zh/canncommercial/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)或[社区版](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)，按照 "选择安装场景" > "准备软件包"（仅驱动和固件）> "安装NPU驱动和固件" 进行安装。

- 请根据NPU型号和CPU架构[拉取IndexSDK镜像](https://www.hiascend.com/developer/ascendhub/detail/7f91c3663b5d4a97b3ae40e3cabbb3a2)，并按照指导运行Index容器。

- 执行`npu-smi info`命令检查驱动是否挂载正常。

## 算子生成

- 以910B4为例（若为其他NPU型号，请参考[此文档](https://gitcode.com/Ascend/IndexSDK/blob/master/docs/zh/user_guide.md#flat)修改），生成512维flat算子：

```bash
cd /usr/local/Ascend/mxIndex/ops && ./custom_opp_*.run
cd /usr/local/Ascend/mxIndex/tools
python3 aicpu_generate_model.py -t 910B4
python3 flat_generate_model.py -d 512 -t 910B4

# 将算子模型移动到MX_INDEX_MODELPATH目录
mv op_models/* $MX_INDEX_MODELPATH
```

## 用例测试

1. 使用Flat（暴力检索）算法进行示例测试：底库大小100万条，特征维度512维，检索向量数128个，TopK为10。创建demo.cpp文件，内容如下：

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

2. 编译demo.cpp：

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

3. 运行demo：

    ```bash
    ./demo

    # 出现以下信息，说明运行成功
    ...
    search end!
    ```
