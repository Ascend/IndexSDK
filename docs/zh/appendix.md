# 附录<a name="ZH-CN_TOPIC_0000001698168789"></a>

## 软件中包含的公网地址<a name="ZH-CN_TOPIC_0000001664167368"></a>

软件中包含的公网地址请参见[Index_SDK_7.3.0_公网地址.xlsx](./resource/Index_SDK_7.3.0_公网地址.xlsx)。


## 环境变量说明<a name="ZH-CN_TOPIC_0000002062005944"></a>

以下环境变量会在程序读取时使用，请勿修改。

**表 1**  环境变量

|环境变量名|说明|
|--|--|
|PATH|可执行程序的文件路径。|
|LD_LIBRARY_PATH|动态链接库路径。|
|PYTHONPATH|python模块文件的默认搜索路径。|
|HOME|当前用户的家目录。|
|PWD|当前系统路径。|
|TMPDIR|临时文件路径。|
|LANG|语言环境。|



## 代码参考<a name="ZH-CN_TOPIC_0000001456375372"></a>

### 简介<a name="ZH-CN_TOPIC_0000001456375408"></a>

手册中提供了在全量和近似的场景下推荐客户所使用的性能最优的Index算法，用户可以参考此章节的参考样例以及Faiss的开源代码开发自己的应用程序，其他算法亦同理。

> [!NOTE] 说明
>-   请注意**AscendFaiss**/**Faiss**需要运行在**try**/**catch**的语句块中，并按照示例中建议的方式进行调用和异常处理。
>-   请确认执行以下代码前已经生成并部署了对应维度的算子。


### AscendIndexSQ<a name="ZH-CN_TOPIC_0000001456694884"></a>

小库算法AscendIndexSQ可以根据一组数据进行训练并生成合适的量化函数，对于输入的float32的特征向量，AscendIndexSQ对其量化为Int8类型的特征向量并存储在Device侧以进一步压缩存储空间，在执行向量比对时，将Int8类型的向量反量化为原始的特征向量执行后续的计算，典型AscendIndexSQ的样例参考如下。

```
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

IndexILFlat为纯Device侧检索方案，利用昇腾AI处理器和AI Core等资源进行各个接口的使能。程序需要在Host侧编译生成二进制文件，然后将二进制文件和相关运行时依赖部署到Device侧。部署请参见下方使用指导，接口使用约束请参考[IndexILFlat](./api/full_retrieval.md#indexilflat接口)。

**参考用例代码<a id="section15454820982"></a>**

```
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

**使用指导<a name="section17204881493"></a>**

1.  IndexILFlat发布件，可在Index SDK软件包安装完成后在安装目录中找到。
    -   头文件：mxIndex/device/include/IndexILFlat.h
    -   动态库：mxIndex/device/lib/libascendfaiss\_minios.so

2.  代码需要使用CANN内置的HCC编译器（默认CANN安装路径下，编译器路径为“/usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++”）进行编译，编译完成后部署到Device侧（具体请参见《CANN 软件安装指南 （开放态,  Atlas 推理系列产品）》的“定制文件系统”章节进行部署）。

    如果需通过SSH服务直接拷贝依赖到Device侧或通过SSH登录到Device上直接运行样例，则需要参考《CANN 软件安装指南 \(开放态,  Atlas 推理系列产品\)》的“使用DSMI接口打开SSH服务”章节解除SSH服务的50M内存占用限制，否则无法发送全部依赖文件，用例无法执行。

3.  算子om文件生成。

    执行以下命令，会在Host侧“mxIndex/modelpath”目录下会生成相关算子文件，算子也需要部署到Device侧。

    ```
    cd mxIndex
    ./ops/custom*
    cd tools
    python3.7 flat_generate_model.py -d 512 --cores 8 -pool 16 -t 310P
    mv op_models/*.om ../modelpath
    ```

4.  编译用例代码。

    在Index SDK工程内新建test路径（mxIndex/test），在test路径下创建“IndexILDemo.cpp”源文件，复制[参考用例代码](#section15454820982)，编译命令参考如下。

    ```
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

5.  部署依赖。
    -   在可执行二进制文件的同级路径下需要配置路径“modelpath”，放置步骤3生成的算子文件。
    -   CANN的相关库依赖，需要将“/usr/local/AscendMiniOs/aarch64-linux/lib64”部署到Device侧，并配置到“LD\_LIBRARY\_PATH”。
    -   将动态库（mxIndex/device/lib/libascendfaiss\_minios.so）部署到Device侧，并配置到环境变量“LD\_LIBRARY\_PATH”。



## 在Device侧运行检索业务<a name="ZH-CN_TOPIC_0000001696207262"></a>

检索目前只包含标准态（即在Host侧运行检索业务），但有的应用场景需要在Device侧运行检索业务，属于一种特定场景。下面介绍在Device侧进行检索的方法。

**前提条件<a name="section178968232301"></a>**

-   已经按照开放态的流程安装CANN，确保“/usr/local/AscendMiniOSRun/”文件夹已存在。具体操作请参见《CANN 软件安装指南 \(开放态, Atlas 推理系列产品\)》。
-   已经解除SSH服务的50M内存占用限制，确保可以发送全部依赖文件。具体操作可参考《CANN 软件安装指南 \(开放态,  Atlas 推理系列产品\)》的“使用DSMI接口打开SSH服务”章节。
-   Host侧需为ARM架构。
-   P2P内存在device侧预留4G，该部分内存默认不可用。若要使用这部分内存，达到最大库容，需使用**npu-smi info set -t p2p-mem-cfg -i "id" -d "value"**命令设置芯片BAR空间拷贝使能状态为“禁用”状态。命令使用可参考《Atlas 中心推理卡 25.3.RC1 npu-smi 命令参考》中的“[设置指定芯片BAR空间拷贝使能状态](https://support.huawei.com/enterprise/zh/doc/EDOC1100523602/dbbc4954)”章节。

**操作步骤<a name="section16775174716308"></a>**

1.  生成要运行的算法所需要的算子。算法介绍请参考[算法介绍](./user_guide.md#算法介绍)。
2.  将以下依赖的库传输到Device侧上。
    -   openblas：/opt/OpenBLAS/lib
    -   Faiss：/usr/local/faiss/faiss1.10.0/lib
    -   运行态toolkit so/usr/local/AscendMiniOSRun/acllib/lib64和/usr/local/AscendMiniOSRun/aarch64-linux/data
    -   检索so：$\{MX\_INDEX\_HOME\}/mxIndex/host/lib，其中\{MX\_INDEX\_HOME\}为Index SDK的安装目录。
    -   Host侧编译器中的libgfortran.so：/usr/lib/aarch64-linux-gnu/libgfortran.so\*
    -   Demo编译出来的二进制
    -   toolkit目录下的latest/opp/version.info文件
    -   算子文件：$\{MX\_INDEX\_HOME\}/modelpath/

        > [!NOTE] 说明 
        >算子文件必须保证只有Atlas 推理系列产品的算子，不能有其他产品的算子，否则可能导致Device侧运行失败。

3.  登录到Device侧，配置以下环境变量。

    ```
    # 配置环境变量
    export LD_LIBRARY_PATH=./lib:./lib64:./
    # 配置version.info文件所在的目录
    export ASCEND_OPP_PATH=./
    ```

4.  登录到Device侧，运行用例。


## 修订记录<a name="ZH-CN_TOPIC_0000001682175202"></a>

|发布日期|修订记录|
|--|--|
|2025-12-30|第一次正式发布|



