# 快速入门

本教程以 [Flat](./05_user_guide.md#全量检索) 索引类型为例，帮助您通过 Docker 容器快速掌握 Index SDK 的基本使用方法。

## 前置条件

开始之前请确认：

- **硬件**：支持 Atlas 推理系列产品、Atlas 800I A2、Atlas 800I A3。可参见“[支持的硬件和操作系统](./01_introduction.md#支持的硬件和操作系统)”。
- **Docker**：已安装并正确配置 Docker 环境，且当前用户可运行容器。
- **Ascend 驱动**：已安装并配置好 Ascend 驱动。

## 步骤 1：拉取镜像

1. **选择匹配版本**
   - 根据当前硬件型号选择对应的镜像版本，注意区分昇腾芯片型号。
   - 根据芯片型号选择对应的镜像拉取命令：

    对于Atlas 推理系列产品，可在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，将查询到的“Name”<a name="npu_type"></a>最后一位数字删除，即是npu_type的取值。
    对于Atlas 800I A2 推理服务器，可在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，查询到的“Name”即是npu_type的取值。
    对于Atlas 800I A3 超节点服务器，可以通过npu-smi info -t board -i 0 -c 0命令进行查询，获取NPU Name信息，910_NPU Name即是npu_type的取值。

   | 芯片型号 | 拉取命令 |
   |---------|---------|
   | 310P | `docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/indexsdk:26.0.0-310p-ubuntu22.04-py3.11` |
   | A3 | `docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/indexsdk:26.0.0-a3-ubuntu22.04-py3.11` |
   | 910b | `docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/indexsdk:26.0.0-910b-ubuntu22.04-py3.11` |

2. **环境预检查**
   - 使用 `npu-smi info` 命令验证 NPU 驱动状态
   - 检查驱动版本与镜像 CANN 版本匹配性（参考《[固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community)》文档，若未安装 NPU 驱动和固件，请先[安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)）

## 步骤 2：运行容器

### 2.1 手动挂载设备

- 设备挂载 ：通过 --device 参数将宿主机的设备文件映射到容器中，确保容器能够访问指定的硬件资源。/dev/davinci为NPU加速卡（按需挂载），/dev/davinci_manager, /dev/devmm_svm， /dev/hisi_hdc为NPU管理设备（全部挂载）。

- 驱动与工具链挂载 ：将宿主机上的驱动文件和工具链目录（如 /usr/local/Ascend/driver 和 /usr/local/bin/npu-smi）以只读方式挂载到容器中，保证容器内的运行环境与宿主机一致。 以下样例代码中，/dev/davinci1 表示挂载 1 号设备。

- -ti后接对应的镜像标签，例如：-it swr.cn-south-1.myhuaweicloud.com/ascendhub/indexsdk:26.0.0-910b-ubuntu22.04-py3.11

```bash
docker run \
    --name index_container \
    --device /dev/davinci1 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -it swr.cn-south-1.myhuaweicloud.com/ascendhub/indexsdk:26.0.0-910b-ubuntu22.04-py3.11 bash

```

### 2.2 容器中执行npu-smi info命令检查驱动是否挂载正常

```bash
npu-smi info #正常显示npu卡信息无报错
```

## 步骤 3：算子生成

- 生成 512 维 flat 算子（`<npu_type>` 取值参考[硬件形态说明](#npu_type)）：

```bash
cd /usr/local/Ascend/mxIndex/ops && ./custom_opp_*.run
cd /usr/local/Ascend/mxIndex/tools
python3 aicpu_generate_model.py -t <npu_type>
python3 flat_generate_model.py -d 512 -t <npu_type>

# 以 Atlas 800I A2服务器上，910B4为例：
# python3 aicpu_generate_model.py -t 910B4
# python3 flat_generate_model.py -d 512 -t 910B4

# MX_INDEX_MODELPATH为存放算子的路径，当前以/home/Ascend/modelpath为例
export MX_INDEX_MODELPATH=/home/Ascend/modelpath
mkdir -p ${MX_INDEX_MODELPATH}
# 将算子模型移动到MX_INDEX_MODELPATH目录
mv op_models/* $MX_INDEX_MODELPATH
```

## 步骤 4：用例测试

1. 使用 Flat算法进行示例测试：底库大小 100 万条，特征维度 512 维，检索向量数 128 个，TopK 为 10。创建 demo.cpp 文件，内容如下：

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
