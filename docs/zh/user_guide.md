# 生成算子<a name="ZH-CN_TOPIC_0000001985832236"></a>

安装Index SDK后，需要依照本章节的指导，设置算子相关的环境变量，并生成算法所需要的算子。

> [!NOTE] 说明
>-   AscendIndexFlat算法L2和IP距离支持在线算子转换，如果环境变量**MX\_INDEX\_USE\_ONLINEOP**设置为1（设置命令：export MX\_INDEX\_USE\_ONLINEOP=1），则会在线转换算子并调用，不需要按照本章节生成离线算子。使用在线算子需要用户在应用程序的最后显式调用<b>\(void\)aclFinalize\(\)</b>（需要包含头文件：\#include "acl/acl.h"）。
>-   对于不支持在线算子的算法，如果设置了环境变量**MX\_INDEX\_USE\_ONLINEOP=1**，会导致程序运行失败。

**操作步骤<a name="section13749124217108"></a>**

1.  进入安装目录“mxIndex-_\{version\}_”，目录及文件名称如[表 Index SDK目录及文件名介绍](#table81133951612)所示。

    ```
    cd mxIndex-{version}
    ```

    **表 1** Index SDK目录及文件名介绍<a id="table81133951612"></a>

    |目录或文件名称|说明|
    |--|--|
    |device|包含IndexIL算法的动态库和头文件。|
    |filelist.txt|软件包文件列表。|
    |host|检索动态库，进行特征检索时，请链接此文件夹下的动态库。|
    |include|API头文件。|
    |lib|检索动态库，链接到host/lib。|
    |modelpath|算子om文件存放目录。编译好算子之后，需要将om文件放置于此文件夹。|
    |ops|包含custom_opp_\<arch>.run脚本，用于检索算法算子安装。|
    |script|包含卸载脚本uninstall.sh，用于卸载Index SDK安装包。|
    |tools|包含用于算子生成python脚本。|
    |version.info|包含版本相关信息。|


2.  进入“ops”目录，编译算子前需要设置“ASCEND\_HOME”、“ASCEND\_VERSION”和“ASCEND\_OPP\_PATH”环境变量，默认分别为\~/Ascend、\~/ascend-toolkit/latest和\~/Ascend/ascend-toolkit/latest/opp。

    ```
    export ASCEND_HOME=~/Ascend
    export ASCEND_VERSION=~/Ascend/ascend-toolkit/latest
    export ASCEND_OPP_PATH=~/Ascend/ascend-toolkit/latest/opp
    ```

    -   “ASCEND\_HOME”表示CANN-toolkit软件安装后文件存储路径。
    -   “ASCEND\_VERSION”表示当前使用的Ascend版本，如果ATC工具安装路径是“/usr/local/Ascend/ascend-toolkit/latest”则无需设置“ASCEND\_HOME”和“ASCEND\_VERSION”。
    -   “ASCEND\_OPP\_PATH”表示算子库根目录，用户需要该目录的写权限。

    > [!NOTE] 说明
    >“MAX\_COMPILE\_CORE\_NUMBER”环境变量用于指定图编译时可用的CPU核数，在算子运行时使用，当前默认为“1”，用户无需设置。

3.  根据实际系统架构执行对应脚本。

    -   ARM架构：

        ```
        ./custom_opp_aarch64.run
        ```

    -   x86\_64架构：

        ```
        ./custom_opp_x86_64.run
        ```

    执行脚本命令时，支持同时输入可选参数，如[表 custom\_opp\__\{arch\}_.run参数说明](#table38211859291)所示。

    **表 2**  custom\_opp\__\{arch\}_.run参数说明<a id="table38211859291"></a>

    |参数名称|说明|
    |--|--|
    |--help \| -h|查询帮助信息。|
    |--info|查询包构建信息。|
    |--list|查询文件列表。|
    |--check|查询包完整性。|
    |--quiet\|-q|可选参数，表示静默安装。减少人机交互的信息的打印。|
    |--nox11|废弃接口，无实际作用。|
    |--noexec|解压软件包到当前目录，但不执行安装脚本。配套--extract=\<path>使用，格式为：--noexec --extract=\<path>。|
    |--extract=\<path>|解压软件包中文件到指定目录。可配套--noexec参数使用。|
    |--tar arg1 [arg2 ...]|对软件包执行tar命令，使用tar后面的参数作为命令的参数。例如执行--tar xvf命令，解压run安装包的内容到当前目录。|


    > [!NOTE] 说明 
    >以下参数未展示在--help参数中，用户请勿直接使用。
    >-   --xwin：使用xwin模式运行。
    >-   --phase2：要求执行第二步动作。

4.  进入“tools”目录，生成所需算子。生成算子之前，需要先确认已经安装CANN的相关依赖。
    -   只生成使用的算法所需要的算子：先参考[算法介绍](#算法介绍)章节，确认算法所需要生成的算子后，再参考[自定义算子介绍](#自定义算子介绍)章节，生成对应的算子。
    -   批量生成所有算法的算子，方法如[表 批量生成算子](#table03891576018)所示。

        **表 3**  批量生成算子<a id="table03891576018"></a>

        |用法|**python3 run_generate_model.py** -m \<mode> -t <npu_type> -p \<pipeline> -pool <pool_size>|
        |--|--|
        |参数名称|\<mode>：算法模式，\<mode>支持ALL以及Flat，SQ8，IVFSQ8，INT8中的一种或多种，多种之间用逗号隔开，如：**python3 run_generate_model.py** **-m Flat,IVFSQ8**。默认全选，可以直接执行**python3 run_generate_model.py**。<br><npu_type>：*npu_type*表示芯片名称。<li>对于Atlas 200/300/500 推理产品、Atlas 推理系列产品，可在安装昇腾AI处理器的服务器执行**npu-smi info**命令进行查询，将查询到的“Name”最后一位数字删除，即是npu_type的取值。<li>对于Atlas 800I A2 推理服务器，可在安装昇腾AI处理器的服务器执行**npu-smi info**命令进行查询，查询到的“Name”即是npu_type的取值。<li>对于Atlas 800I A3 超节点服务器，可以通过**npu-smi info -t board -i 0 -c 0**命令进行查询，获取**NPU Name**信息，910_**NPU Name**即是npu_type的取值。<br>\<pipeline>：是否使用多线程并行流水生成算子模型，默认为true。设置为true时，使用默认的pool_size的值为32。<br><pool_size>：批量生成算子多进程调度的进程池大小。<br>--help \| -h：查询帮助信息。|
        |说明| <li>执行此命令，用户可以得到多组算子模型文件。执行命令前，用户需要更改当前目录下的para_table.xml文件，将所需的参数填入表中。<li>1 ≤ pool_size ≤ 32|


        > [!NOTE] 说明 
        >算子生成说明表格中的约束说明，代表业务中经常涉及的参数组合，使用其他参数运行异常请参见《CANN ATC离线模型编译工具用户指南》。

5.  准备算子模型文件。

    -   可以将算子模型文件目录配置为环境变量“MX\_INDEX\_MODELPATH”（环境变量支持以\~开头的路径、相对路径和绝对路径，**路径中不能包含软链接**；使用该变量时将统一转化为绝对路径并限制在“/home”或“/root”路径下）。

        ```
        mv op_models/* $PWD/../modelpath
        export MX_INDEX_MODELPATH=`realpath $PWD/../modelpath`
        ```

    -   如未使用环境变量进行配置，需将算子模型文件移动到当前目录的“modelpath”目录下。

    算子生成后，请妥善保管相关om文件并确保文件不被篡改。

    > [!NOTE] 说明
    >生成算子时如果出现报错：Failed to import Python module，请参照[NumPy的数据类型np.float\_ 已被移除](./faq.md#numpy的数据类型npfloat_-已被移除)解决。

# 使用样例<a name="ZH-CN_TOPIC_0000001649689164"></a>

本章节提供了一个简单的样例，帮助用户快速体验运用Index SDK进行检索的流程。

假定在Atlas 推理系列产品上，有业务需要使用到暴搜（Flat）算法，底库大小为100w，维度是512维，需要检索的向量是128个，topk是10，编写一个Demo调用Index接口大致步骤如下。

**前提条件<a name="section42712132135"></a>**

-   已完成[安装部署](./installation_guide.md#安装部署)操作。
-   已经生成Flat和AICPU算子。

**操作步骤<a name="section1592313504162"></a>**

1.  构造Demo，过程包括：

    1.  Demo中引入暴搜（Flat）的头文件。
    2.  构造底库向量数据，这里用随机数生成代替。
    3.  归一化底库数据。
    4.  初始化Flat的Index。
    5.  调用接口添加底库。
    6.  调用接口进行检索。

    demo.cpp代码如下：

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

2.  编译demo.cpp

    ```
    # 以安装路径“/home/work/FeatureRetrieval”为例
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

3.  运行Demo，显示search end!即表示Demo运行成功。

    ```
    ./demo
    ...
    search end!
    ```


# 算法介绍<a name="ZH-CN_TOPIC_0000001649848468"></a>

> [!NOTE] 说明 
>标准态部署主要使用AI CPU，Ctrl CPU和AI CPU的最佳推荐配比如下。
>-   使用Atlas 200/300/500 推理产品，建议设置为2:6。
>-   使用Atlas 推理系列产品，建议设置为1:7。

## 全量检索<a name="ZH-CN_TOPIC_0000001698088061"></a>

**全量检索算法介绍<a name="section46312418528"></a>**

|算法（API参考）|算法使用场景|需要生成的算子|样例链接|
|--|--|--|--|
|[AscendIndexInt8Flat](./api/full_retrieval.md#ascendindexint8flat)|<li>特征类型：int8<li>特征维度：64, 128, 256, 384, 512, 768, 1024<li>距离类型：L2和IP<li>计算精度：高<li>Device内存占用：较低<li>适应场景：精度要求高的暴力检索场景|<li>[INT8Flat](#int8flat)<li>[AICPU](#aicpu)|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexInt8Flat.cpp">链接</a>|
|[AscendIndexFlat](./api/full_retrieval.md#ascendindexflat)|<li>特征类型：FP32、FP16<li>特征维度：32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096<li>距离类型：L2和IP<li>计算精度：高<li>Device内存占用：高<li>适应场景：精度要求高的暴力检索场景；IP距离推荐在dim > 128的场景下使用。|<li>[Flat](#flat)<li>[AICPU](#aicpu)|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexFlat.cpp">链接</a>|
|[AscendIndexSQ](./api/full_retrieval.md#ascendindexsq)|<li>特征类型：FP32<li>特征维度：64, 128, 256, 384, 512, 768<li>距离类型：L2和IP<li>计算精度：高<li>Device内存占用：较低（已量化为int8）<li>适应场景：精度要求较高的暴力检索场景|<li>[SQ8](#sq8)<li>[AICPU](#aicpu)|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexSQ.cpp">链接</a>|
|[AscendIndexCluster](./api/full_retrieval.md#ascendindexcluster)|<li>特征类型：FP32<li>特征维度：32, 64, 128, 256, 384, 512<li>距离类型：IP<li>计算精度：高<li>Device内存占用：较高<li>适应场景：只计算距离的聚类场景<li>仅支持Atlas 推理系列产品|<li>[Flat](#flat)<li>[AICPU](#aicpu)|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexCluster.cpp">链接</a>|
|[IndexIL](./api/full_retrieval.md#indexil)|需要运行在Device上，安装部署复杂，暂不推荐使用|<li>[Flat](#flat)|参考[IndexILFlat](./api/full_retrieval.md#indexilflat)|
|[AscendIndexILFlat](./api/full_retrieval.md#ascendindexilflat)|<li>特征类型：FP16、FP32<li>特征维度：32, 64, 128, 256, 384, 512<li>距离类型：IP<li>计算精度：高<li>Device内存占用：较高<li>适应场景：只计算距离的聚类场景<li>仅支持Atlas 推理系列产品|<li>[Flat](#flat)<li>[AICPU](#aicpu)|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/tree/master/IndexSDK">链接</a>|



## 近似检索<a name="ZH-CN_TOPIC_0000001698168797"></a>

**近似检索算法介绍<a name="section46312418528"></a>**

|算法（API参考）|算法使用场景|需要生成的算子|样例链接|
|--|--|--|--|
|[AscendIndexIVFSP](./api/approximate_retrieval.md#ascendindexivfsp)|<li>特征类型：FP32<li>特征维度：64, 128, 256, 512, 768<li>距离类型：L2<li>计算精度：中<li>Device内存占用：低（压缩特征）<li>适应场景：适用于亿级底库（大库容），对性能要求较高，对精度损失有容忍的近似检索场景。<li>仅支持Atlas 推理系列产品|<li>IVFSP业务算子<li>IVFSP AICPU算子<li>IVFSP训练算子（仅在需要通过训练生成码本文件时才使用到）<br>请参见[IVFSP](#ivfsp)。|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFSP.cpp">链接</a>|
|[AscendIndexIVFSQ](./api/approximate_retrieval.md#ascendindexivfsq)|<li>特征类型：FP32<li>特征维度：64, 128, 256, 384, 512<li>距离类型：L2和IP<li>计算精度：中<li>Device内存占用：较低（量化为int8）<li>适应场景：IVFSQ算法作为性能-精度调节器，适用于对精度损失有容忍，但是对性能要求比较高的场景。|<li>[IVFSQ8](#ivfsq8)<li>[AICPU](#aicpu)<li>[FlatAT](#flatat)（仅在参数useKmeansPP设置为true的时候需要生成FlatAT算子）|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFSQ.cpp">链接</a>|
|[AscendIndexIVFSQT](./api/approximate_retrieval.md#ascendindexivfsqt)|<li>特征类型：FP32<li>特征维度：256<li>距离类型：IP<li>计算精度：中<li>Device内存占用：低（量化和降维）<li>适应场景：AscendIndexIVFSQT包含降维算法的三级检索IVFSQ算法，适用于亿级底库（大库容），对性能要求较高，对精度损失有容忍的近似检索场景。|<li>[IVFSQT](#ivfsqt)<li>[FlatAT](#flatat)<li>[AICPU](#aicpu)<li>[FlatInt8AT](#flatint8at)（在Atlas 推理系列产品上时需要生成）|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFSQT.cpp">链接</a>|
|[AscendIndexBinaryFlat](./api/approximate_retrieval.md#ascendindexbinaryflat)|<li>特征类型：uint8二值化特征<li>特征维度：256, 512, 1024<li>距离类型：Hamming和IP<li>计算精度：高<li>Device内存占用：低<li>适应场景：AscendIndexBinaryFlat类继承自Faiss的IndexBinary类，用于二值化特征检索。对内存占用要求较低，性能要求较高的场景。<li>仅支持Atlas 推理系列产品|<li>[BinaryFlat](#binaryflat)<li>[AICPU](#aicpu)|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexBinaryFlat.cpp">链接</a>|
|[AscendIndexVStar](./api/approximate_retrieval.md#ascendindexvstar)|<li>特征类型：FP32<li>特征维度：128, 256, 512, 1024<li>距离类型：L2<li>计算精度：中<li>Device内存占用：低（压缩特征）<li>适应场景：适用于千万级底库（大库容），对性能要求较高，对精度损失有容忍的近似检索场景。<li>仅支持Atlas 推理系列产品|<li>VStar业务算子<li>VStar AICPU算子<li>VStar训练算子（仅在需要通过训练生成码本文件时才使用到）<br>请参见[VSTAR](#vstar)。|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexVStar.cpp">链接</a>|
|[AscendIndexGreat](./api/approximate_retrieval.md#ascendindexgreat)|<li>特征类型：FP32<li>特征维度：128, 256, 512, 1024<li>距离类型：L2<li>计算精度：中<li>Device内存占用：低（压缩特征）<li>适应场景：适用于千万级底库（大库容），对性能要求较高，对精度损失有容忍的近似检索场景。<li>仅支持Atlas 推理系列产品|（当mode为AKMode时，才需要生成算子）<li>VStar业务算子<li>VStar AICPU算子<li>VStar训练算子（仅在需要通过训练生成码本文件时才使用到）<br>请参见[VSTAR](#vstar)。|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexGreat.cpp">链接</a>|
|[AscendIndexIVFFlat](./api/approximate_retrieval.md#ascendindexivfflat)|<li>特征类型：FP32<li>特征维度：128<li>距离类型：IP<li>计算精度：中<li>Device内存占用：低（压缩特征）<li>适应场景：适用于亿级底库（大库容），对性能要求较高，对精度损失有容忍的近似检索场景。<li>仅支持Atlas A2 推理系列产品和Atlas A3 推理系列产品|<li>[AICPU](#aicpu)<li>[IVFFLAT](#ivfflat)|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFFlat.cpp">链接</a>|



## 属性过滤检索<a name="ZH-CN_TOPIC_0000001649689168"></a>

**属性过滤检索算法介绍<a name="section46312418528"></a>**

|算法（API参考）|算法使用场景|需要生成的算子|样例链接|
|--|--|--|--|
|[AscendIndexTS](./api/attribute_filtering-based_retrieva.md#ascendindexts)|<li>特征类型：uint8二值化特征、int8、FP32（具体算法不同而不同）<li>特征维度：具体算法不同而不同<li>距离类型：Hamming、Cos、IP、L2<li>计算精度：较高<li>Device内存占用：较高<li>适应场景：需要过滤属性的时空库场景<li>Hamming距离仅支持Atlas 推理系列产品|<li>[Mask](#mask)<li>[BinaryFlat](#binaryflat)<li>[Int8Flat](#int8flat)<li>[Flat](#flat)<li>[AICPU](#aicpu)|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexTS.cpp">链接</a>|



## 多Index批量检索<a name="ZH-CN_TOPIC_0000001649848472"></a>

**接口介绍<a name="section46312418528"></a>**

|接口（API参考）|**接口使用场景**|**可以使用本接口的算法**|样例链接|
|--|--|--|--|
|[Search](./api/multi-index_batch_retrieval.md#searchfaissindex接口)|单Device进行多个Index检索。|<li>[AscendIndexSQ](./api/full_retrieval.md#ascendindexsq)<li>[AscendIndexFlat](./api/full_retrieval.md#ascendindexflat)<li>[AscendIndexIVFSP](./api/approximate_retrieval.md#ascendindexivfsp)|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">链接</a>|
|[Search](./api/multi-index_batch_retrieval.md#searchascendindex接口)|单Device进行多个AscendIndex检索。|<li>[AscendIndexSQ](./api/full_retrieval.md#ascendindexsq)<li>[AscendIndexFlat](./api/full_retrieval.md#ascendindexflat)<li>[AscendIndexIVFSP](./api/approximate_retrieval.md#ascendindexivfsp)|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">链接</a>|
|[Search](./api/multi-index_batch_retrieval.md#searchascendindexint8接口)|单Device进行多个AscendIndexInt8检索。|<li>[AscendIndexInt8Flat](./api/full_retrieval.md#ascendindexint8flat)|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">链接</a>|
|[SearchWithFilter](./api/multi-index_batch_retrieval.md#searchwithfilterfaissindex单filter接口)|单Device进行多个Index带属性过滤（单filter）检索。|<li>[AscendIndexSQ](./api/full_retrieval.md#ascendindexsq)<li>[AscendIndexIVFSP](./api/approximate_retrieval.md#ascendindexivfsp)|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">链接</a>|
|[SearchWithFilter](./api/multi-index_batch_retrieval.md#searchwithfilterascendindex单filter接口)|单Device进行多个AscendIndex带属性过滤（单filter）检索。|<li>[AscendIndexSQ](./api/full_retrieval.md#ascendindexsq)<li>[AscendIndexIVFSP](./api/approximate_retrieval.md#ascendindexivfsp)|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">链接</a>|
|[SearchWithFilter](./api/multi-index_batch_retrieval.md#searchwithfilterfaissindex多filter接口)|单Device进行多个Index带过滤属性（多filter）检索。|<li>[AscendIndexSQ](./api/full_retrieval.md#ascendindexsq)<li>[AscendIndexIVFSP](./api/approximate_retrieval.md#ascendindexivfsp)|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">链接</a>|
|[SearchWithFilter](./api/multi-index_batch_retrieval.md#searchwithfilterascendindex多filter接口)|单Device进行多个AscendIndex带过滤属性（多filter）检索。|<li>[AscendIndexSQ](./api/full_retrieval.md#ascendindexsq)<li>[AscendIndexIVFSP](./api/approximate_retrieval.md#ascendindexivfsp)|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">链接</a>|



## 其他功能<a name="ZH-CN_TOPIC_0000001698088065"></a>

**算法介绍<a name="section46312418528"></a>**

|算法（API参考）|算法需求（性能、场景差异）|如何调用|样例链接|
|--|--|--|--|
|[IReduction](./api/more_functions.md#ireduction)|IReduction是特征检索组件中降维方法的统一接口，目前支持**PCAR**和**NN**两种降维算法。|通过ReductionConfig初始化，调用CreateReduction创建降维对象，然后进行train和reduce。|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIReduction.cpp">链接</a>|
|[AscendNNInference](./api/more_functions.md#ascendnninference)|通过神经网络进行推理。|通过AscendNNInference创建NN降维对象，然后进行infer降维。|<a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIReduction.cpp">链接</a>|
|[AscendCloner](./api/more_functions.md#ascendcloner)|Index SDK提供了将NPU上的检索Index资源拷贝到CPU侧Faiss的操作，拷贝过程发生在内存中，原始NPU的Index上加载的数据会被拷贝到CPU侧的内存中，方便用户在CPU上使用相同的底库执行检索。|index_ascend_to_cpu将AscendIndex拷贝生成一个CPU上的Index，index_cpu_to_ascend将CPU上的Index拷贝生成一个AscendIndex。|无|


# 自定义算子介绍<a name="ZH-CN_TOPIC_0000001456854988"></a>

## 自定义算子简介<a name="ZH-CN_TOPIC_0000001456695000"></a>

特征检索方案使用TIK算子开发实现特征距离计算逻辑，包含以下的自定义算子。

-   [Flat距离计算算子](#flat)：得到特征底库数据和待检索的特征向量之间的距离（L2/IP）。
-   [SQ8距离计算算子](#sq8)：得到SQ量化的特征底库数据和待检索的未量化特征向量之间的距离（L2/IP）。
-   [IVFSQ8算子](#ivfsq8)：得到IVFSQ8算法所需要的算子。
-   [INT8Flat距离计算算子](#int8flat)：得到INT8量化的特征底库数据和待检索的INT8量化特征向量之间的距离（L2/COS）。
-   [IVFSQT算子](#ivfsqt)：得到IVFSQT算法一二三级所需的距离算子。
-   [FlatAT算子](#flatat)：主要用于在IVF场景，减少train和add的耗时，其中“code\_num”等于“nlist”。
-   [FlatInt8AT算子](#flatint8at)：优化在Atlas 推理系列产品下IVFSQT中train、add与update的耗时。
-   [AICPU算子](#aicpu)：调度昇腾AI处理器的CPU完成排序等计算，充分利用硬件性能。
-   [BinaryFlat算子](#binaryflat)：得到二值化算法所需算子。
-   [Mask算子](#mask)：得到时空库属性过滤算法所需的Mask算子。
-   [IVFSP算子](#ivfsp)：得到IVFSP算法所需的业务算子、AICPU算子，以及训练生成IVFSP码本时所需的训练算子。
-   [VStar算子](#vstar)：得到VStar算法所需的业务算子、AICPU算子。
-   [IVFFLAT](#ivfflat)：得到IVFFLAT算法一级二级所需的距离算子。


## 算子生成说明<a name="ZH-CN_TOPIC_0000001456695052"></a>

### Flat<a name="ZH-CN_TOPIC_0000001506495813"></a>

<a name="table3955133174816"></a>
<table><tbody><tr id="row3956113304810"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p1995613338481"><a name="p1995613338481"></a><a name="p1995613338481"></a>用法</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p151131757175117"><a name="p151131757175117"></a><a name="p151131757175117"></a>python3 flat_generate_model.py -d &lt;dim&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row1695612338483"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p395693310480"><a name="p395693310480"></a><a name="p395693310480"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p1242365818554"><a name="p1242365818554"></a><a name="p1242365818554"></a>&lt;dim&gt;：特征向量维度D，默认值为“512”。</p>
<p id="p19895115711230"><a name="p19895115711230"></a><a name="p19895115711230"></a>&lt;core_num&gt;：<span id="ph129021410310"><a name="ph129021410310"></a><a name="ph129021410310"></a>昇腾AI处理器</span>AI Core的个数，默认值为<span class="parmvalue" id="parmvalue136315104287"><a name="parmvalue136315104287"></a><a name="parmvalue136315104287"></a>“2”</span>。无需设置。</p>
<p id="p1489519612244"><a name="p1489519612244"></a><a name="p1489519612244"></a>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为<span class="parmvalue" id="parmvalue12408135012357"><a name="parmvalue12408135012357"></a><a name="parmvalue12408135012357"></a>“0”</span>，无需设置。</p>
<p id="p15972135916413"><a name="p15972135916413"></a><a name="p15972135916413"></a>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认值为<span class="parmvalue" id="parmvalue771714711233"><a name="parmvalue771714711233"></a><a name="parmvalue771714711233"></a>“10”</span>。</p>
<p id="p16833155612195"><a name="p16833155612195"></a><a name="p16833155612195"></a>&lt;npu_type&gt;：硬件形态。</p>
<a name="ul994471125418"></a><a name="ul994471125418"></a><ul id="ul994471125418"><li><span id="ph10941163375016"><a name="ph10941163375016"></a><a name="ph10941163375016"></a>对于<span id="ph1294133365010"><a name="ph1294133365010"></a><a name="ph1294133365010"></a><term id="zh-cn_topic_0000001519959665_term169221139190"><a name="zh-cn_topic_0000001519959665_term169221139190"></a><a name="zh-cn_topic_0000001519959665_term169221139190"></a>Atlas 200/300/500 推理产品</term></span>、<span id="ph19941183375011"><a name="ph19941183375011"></a><a name="ph19941183375011"></a>Atlas 推理系列产品</span>，可在安装昇腾AI处理器的服务器执行<strong id="b7330834135115"><a name="b7330834135115"></a><a name="b7330834135115"></a>npu-smi info</strong>命令进行查询，将查询到的“Name”最后一位数字删除，即是npu_type的取值</span>。</li><li>对于<span id="ph299603920504"><a name="ph299603920504"></a><a name="ph299603920504"></a>Atlas 800I A2 推理服务器</span>，<span id="ph18599444165014"><a name="ph18599444165014"></a><a name="ph18599444165014"></a>可在安装昇腾AI处理器的服务器执行<strong id="b18495139195117"><a name="b18495139195117"></a><a name="b18495139195117"></a>npu-smi info</strong>命令进行查询，查询到的“Name”即是npu_type的取值</span>。</li><li>对于<span id="ph6488102065112"><a name="ph6488102065112"></a><a name="ph6488102065112"></a>Atlas 800I A3 超节点服务器</span>，可以通过<strong id="b1248815206511"><a name="b1248815206511"></a><a name="b1248815206511"></a>npu-smi info -t board -i 0 -c 0</strong>命令进行查询，获取<strong id="b144882206516"><a name="b144882206516"></a><a name="b144882206516"></a>NPU Name</strong>信息，910_<strong id="b1648872017519"><a name="b1648872017519"></a><a name="b1648872017519"></a>NPU Name</strong>即是npu_type的取值。</li></ul>
<p id="p952414873216"><a name="p952414873216"></a><a name="p952414873216"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row15956133317485"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p11956183311486"><a name="p11956183311486"></a><a name="p11956183311486"></a>说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p9956933114816"><a name="p9956933114816"></a><a name="p9956933114816"></a>执行此命令，用户可以得到一组距离计算算子模型文件，用户需要自行修改命令中的参数。</p>
</td>
</tr>
<tr id="row3636101012016"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p176373101504"><a name="p176373101504"></a><a name="p176373101504"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul9805193810583"></a><a name="ul9805193810583"></a><ul id="ul9805193810583"><li>dim ∈ {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}</li><li>0 ≤ pool_size ≤ 32</li></ul>
</td>
</tr>
</tbody>
</table>

**涉及算法<a name="section1467921619472"></a>**

-   [AscendIndexFlat](#全量检索)
-   [AscendIndexCluster](#全量检索)
-   [IndexIL](#全量检索)
-   [AscendIndexTS](#属性过滤检索)
-   [Search（单device进行多个Index检索）](./api/multi-index_batch_retrieval.md#searchfaissindex接口)
-   [Search（单device进行多个AscendIndex检索）](./api/multi-index_batch_retrieval.md#searchascendindex接口)


### SQ8<a name="ZH-CN_TOPIC_0000001506614921"></a>

> [!NOTE] 说明
>INT8Flat和SQ8的区别主要在于：INT8由外部进行量化，Index的输入特征是INT8类型，SQ8由Index内部量化，Index的输入特征是Float32类型。

<a name="table3955133174816"></a>
<table><tbody><tr id="row3956113304810"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p1995613338481"><a name="p1995613338481"></a><a name="p1995613338481"></a>用法</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p185151728356"><a name="p185151728356"></a><a name="p185151728356"></a>python3 sq8_generate_model.py -d &lt;dim&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row1695612338483"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p395693310480"><a name="p395693310480"></a><a name="p395693310480"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p142073616319"><a name="p142073616319"></a><a name="p142073616319"></a>&lt;dim&gt;：特征向量维度D，默认值为“128”。</p>
<p id="p76412712341"><a name="p76412712341"></a><a name="p76412712341"></a>&lt;core_num&gt;：<span id="ph129021410310"><a name="ph129021410310"></a><a name="ph129021410310"></a>昇腾AI处理器</span>AI Core的个数，默认为<span class="parmvalue" id="parmvalue172734465517"><a name="parmvalue172734465517"></a><a name="parmvalue172734465517"></a>“2”</span>。不指定该值时，根据&lt;npu_type&gt;配置：当npu_type配置为310时，&lt;core_num&gt;配置为2；当npu_type配置为310P时，&lt;core_num&gt;配置为8。</p>
<p id="p882511475345"><a name="p882511475345"></a><a name="p882511475345"></a>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为<span class="parmvalue" id="parmvalue12408135012357"><a name="parmvalue12408135012357"></a><a name="parmvalue12408135012357"></a>“0”</span>，无需设置。</p>
<p id="p822871512241"><a name="p822871512241"></a><a name="p822871512241"></a>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认值为<span class="parmvalue" id="parmvalue771714711233"><a name="parmvalue771714711233"></a><a name="parmvalue771714711233"></a>“10”</span>。</p>
<p id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持<span id="ph97104582114"><a name="ph97104582114"></a><a name="ph97104582114"></a><term id="zh-cn_topic_0000001519959665_term169221139190"><a name="zh-cn_topic_0000001519959665_term169221139190"></a><a name="zh-cn_topic_0000001519959665_term169221139190"></a>Atlas 200/300/500 推理产品</term></span>以及<span id="ph19590185162111"><a name="ph19590185162111"></a><a name="ph19590185162111"></a>Atlas 推理系列产品</span>，取值范围分别为：310、310P，默认为<span class="parmvalue" id="parmvalue68401116171220"><a name="parmvalue68401116171220"></a><a name="parmvalue68401116171220"></a>“310”</span>。</p>
<p id="p1692503012329"><a name="p1692503012329"></a><a name="p1692503012329"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row15956133317485"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p11956183311486"><a name="p11956183311486"></a><a name="p11956183311486"></a>说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p158811120735"><a name="p158811120735"></a><a name="p158811120735"></a>执行此命令，用户可以得到一组SQ8距离计算算子模型文件，用户需要自行修改命令中的参数。</p>
</td>
</tr>
<tr id="row1080311205318"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p280411209314"><a name="p280411209314"></a><a name="p280411209314"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul1361419421018"></a><a name="ul1361419421018"></a><ul id="ul1361419421018"><li>dim ∈ {64, 128, 256, 384, 512, 768}</li><li>0 ≤ pool_size ≤ 32</li></ul>
</td>
</tr>
</tbody>
</table>

**涉及算法<a name="section6413836184719"></a>**

-   [AscendIndexSQ](#全量检索)
-   [Search（单device进行多个Index检索）](./api/multi-index_batch_retrieval.md#searchfaissindex接口)
-   [Search（单device进行多个AscendIndex检索）](./api/multi-index_batch_retrieval.md#searchascendindex接口)
-   [SearchWithFilter（FaissIndex单filter）](./api/multi-index_batch_retrieval.md#searchwithfilterfaissindex单filter接口)
-   [SearchWithFilter（AscendIndex单filter）](./api/multi-index_batch_retrieval.md#searchwithfilterascendindex单filter接口)
-   [SearchWithFilter（FaissIndex多filter）](./api/multi-index_batch_retrieval.md#searchwithfilterfaissindex多filter接口)
-   [SearchWithFilter（AscendIndex多filter）](./api/multi-index_batch_retrieval.md#searchwithfilterascendindex多filter接口)


### IVFSQ8<a name="ZH-CN_TOPIC_0000001506614889"></a>

<a name="table3955133174816"></a>
<table><tbody><tr id="row3956113304810"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p1995613338481"><a name="p1995613338481"></a><a name="p1995613338481"></a>用法</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p151131757175117"><a name="p151131757175117"></a><a name="p151131757175117"></a>python3 ivfsq8_generate_model.py -d &lt;dim&gt; -c &lt;coarse_centroid_num&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row1695612338483"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p395693310480"><a name="p395693310480"></a><a name="p395693310480"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p114841371610"><a name="p114841371610"></a><a name="p114841371610"></a>&lt;dim&gt;：特征向量维度D，默认值为“128”。</p>
<p id="p1157915132617"><a name="p1157915132617"></a><a name="p1157915132617"></a>&lt;coarse_centroid_num&gt;：L1簇聚类中心个数，默认值为“16384”。</p>
<p id="p45741834183717"><a name="p45741834183717"></a><a name="p45741834183717"></a>&lt;core_num&gt;：<span id="ph129021410310"><a name="ph129021410310"></a><a name="ph129021410310"></a>昇腾AI处理器</span>AI Core的个数，默认为<span class="parmvalue" id="parmvalue76481515195212"><a name="parmvalue76481515195212"></a><a name="parmvalue76481515195212"></a>“2”</span>。不指定该值时，根据&lt;npu_type&gt;配置：当npu_type配置为310时，&lt;core_num&gt;配置为2；当npu_type配置为310P时，&lt;core_num&gt;配置为8。</p>
<p id="p14268135033720"><a name="p14268135033720"></a><a name="p14268135033720"></a>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为<span class="parmvalue" id="parmvalue12408135012357"><a name="parmvalue12408135012357"></a><a name="parmvalue12408135012357"></a>“0”</span>，无需设置。</p>
<p id="p17428219182417"><a name="p17428219182417"></a><a name="p17428219182417"></a>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认值为<span class="parmvalue" id="parmvalue771714711233"><a name="parmvalue771714711233"></a><a name="parmvalue771714711233"></a>“10”</span>。</p>
<p id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持<span id="ph97104582114"><a name="ph97104582114"></a><a name="ph97104582114"></a><term id="zh-cn_topic_0000001519959665_term169221139190"><a name="zh-cn_topic_0000001519959665_term169221139190"></a><a name="zh-cn_topic_0000001519959665_term169221139190"></a>Atlas 200/300/500 推理产品</term></span>以及<span id="ph19590185162111"><a name="ph19590185162111"></a><a name="ph19590185162111"></a>Atlas 推理系列产品</span>，取值范围分别为：310、310P，默认为<span class="parmvalue" id="parmvalue68401116171220"><a name="parmvalue68401116171220"></a><a name="parmvalue68401116171220"></a>“310”</span>。</p>
<p id="p824473918329"><a name="p824473918329"></a><a name="p824473918329"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row15956133317485"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p11956183311486"><a name="p11956183311486"></a><a name="p11956183311486"></a>说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p1648417714614"><a name="p1648417714614"></a><a name="p1648417714614"></a>执行此命令，用户可以得到一组算子模型文件，用户需要自行修改命令中的参数。</p>
</td>
</tr>
<tr id="row2657434476"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p1565893415718"><a name="p1565893415718"></a><a name="p1565893415718"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul1186212111343"></a><a name="ul1186212111343"></a><ul id="ul1186212111343"><li>dim ∈ {64, 128, 256, 384, 512}</li><li>coarse centroid num ∈ {1024, 2048, 4096, 8192, 16384, 32768}</li><li>0 ≤ pool_size ≤ 32</li></ul>
</td>
</tr>
</tbody>
</table>

**涉及算法<a name="section14565105918474"></a>**

[AscendIndexIVFSQ](#近似检索)


### INT8Flat<a name="ZH-CN_TOPIC_0000001456695008"></a>

> [!NOTE] 说明 
>INT8Flat和SQ8的区别主要在于：INT8由外部进行量化，Index的输入特征是INT8类型，SQ8由Index内部量化，Index的输入特征是Float32类型。

<a name="table3955133174816"></a>
<table><tbody><tr id="row3956113304810"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p1995613338481"><a name="p1995613338481"></a><a name="p1995613338481"></a>用法</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p126011312484"><a name="p126011312484"></a><a name="p126011312484"></a>python3 int8flat_generate_model.py -d &lt;dim&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt; -code &lt;code_num&gt;</p>
</td>
</tr>
<tr id="row1695612338483"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p395693310480"><a name="p395693310480"></a><a name="p395693310480"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p114841371610"><a name="p114841371610"></a><a name="p114841371610"></a>&lt;dim&gt;：特征向量维度D，默认值为“512”。</p>
<p id="p76412712341"><a name="p76412712341"></a><a name="p76412712341"></a>&lt;core_num&gt;：<span id="ph129021410310"><a name="ph129021410310"></a><a name="ph129021410310"></a>昇腾AI处理器</span>AI Core的个数，默认为<span class="parmvalue" id="parmvalue1790915218539"><a name="parmvalue1790915218539"></a><a name="parmvalue1790915218539"></a>“2”</span>。无需设置。</p>
<p id="p1489519612244"><a name="p1489519612244"></a><a name="p1489519612244"></a>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为<span class="parmvalue" id="parmvalue12408135012357"><a name="parmvalue12408135012357"></a><a name="parmvalue12408135012357"></a>“0”</span>，无需设置。</p>
<p id="p67082264240"><a name="p67082264240"></a><a name="p67082264240"></a>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认值为<span class="parmvalue" id="parmvalue771714711233"><a name="parmvalue771714711233"></a><a name="parmvalue771714711233"></a>“10”</span>。</p>
<div class="p" id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;：硬件形态。<a name="ul10641721165519"></a><a name="ul10641721165519"></a><ul id="ul10641721165519"><li><span id="ph10941163375016"><a name="ph10941163375016"></a><a name="ph10941163375016"></a>对于<span id="ph1294133365010"><a name="ph1294133365010"></a><a name="ph1294133365010"></a><term id="zh-cn_topic_0000001519959665_term169221139190"><a name="zh-cn_topic_0000001519959665_term169221139190"></a><a name="zh-cn_topic_0000001519959665_term169221139190"></a>Atlas 200/300/500 推理产品</term></span>、<span id="ph19941183375011"><a name="ph19941183375011"></a><a name="ph19941183375011"></a>Atlas 推理系列产品</span>，可在安装昇腾AI处理器的服务器执行<strong id="b7330834135115"><a name="b7330834135115"></a><a name="b7330834135115"></a>npu-smi info</strong>命令进行查询，将查询到的“Name”最后一位数字删除，即是npu_type的取值</span>。</li><li>对于<span id="ph299603920504"><a name="ph299603920504"></a><a name="ph299603920504"></a>Atlas 800I A2 推理服务器</span>，<span id="ph18599444165014"><a name="ph18599444165014"></a><a name="ph18599444165014"></a>可在安装昇腾AI处理器的服务器执行<strong id="b18495139195117"><a name="b18495139195117"></a><a name="b18495139195117"></a>npu-smi info</strong>命令进行查询，查询到的“Name”即是npu_type的取值</span>。</li></ul>
</div>
<p id="p6501256288"><a name="p6501256288"></a><a name="p6501256288"></a>&lt;code_num&gt;：算子调用时底库分块大小，默认值为“262144”，不设置时默认生成所有code_num值的算子。</p>
<p id="p11599745183215"><a name="p11599745183215"></a><a name="p11599745183215"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row15956133317485"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p11956183311486"><a name="p11956183311486"></a><a name="p11956183311486"></a>说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p1648417714614"><a name="p1648417714614"></a><a name="p1648417714614"></a>执行此命令，用户可以得到一组算子模型文件，用户需要自行修改命令中的参数。</p>
</td>
</tr>
<tr id="row13262151218181"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p1552013434810"><a name="p1552013434810"></a><a name="p1552013434810"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul13923029345"></a><a name="ul13923029345"></a><ul id="ul13923029345"><li>dim ∈ {64, 128, 256, 384, 512, 768, 1024}</li><li>0 ≤ pool_size ≤ 32</li><li>code_num ∈ {16384, 32768, 65536, 131072, 262144}</li></ul>
</td>
</tr>
</tbody>
</table>

**涉及算法<a name="section3261111214818"></a>**

-   [AscendIndexInt8Flat](#全量检索)
-   [AscendIndexTS](#属性过滤检索)
-   [Search（单device进行多个AscendIndexInt8检索）](./api/multi-index_batch_retrieval.md#searchascendindexint8接口)


### IVFSQT<a name="ZH-CN_TOPIC_0000001506414677"></a>

> [!NOTE] 说明
>为了减少train和add的耗时，需要生成FlatAT算子。其中，Flat的<dim\>需与IVFSQT的<dim\_in\>相同，Flat的<code\_num\>与IVFSQT的<coarse\_centroid\_num\>一致。

<a name="table3955133174816"></a>
<table><tbody><tr id="row3956113304810"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p1995613338481"><a name="p1995613338481"></a><a name="p1995613338481"></a>用法</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p151131757175117"><a name="p151131757175117"></a><a name="p151131757175117"></a>python3 ivfsqt_generate_model.py --cores &lt;core_num&gt; -d &lt;dim_in&gt; -r &lt;compress_ratio&gt; -c &lt;coarse_centroid_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row1695612338483"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p395693310480"><a name="p395693310480"></a><a name="p395693310480"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p114841371610"><a name="p114841371610"></a><a name="p114841371610"></a>&lt;dim_in&gt;：输入特征向量维度，默认值为“256”。</p>
<p id="p334923265916"><a name="p334923265916"></a><a name="p334923265916"></a>&lt;compress_ratio&gt;：输入与输出维度的比值，默认值为<span class="parmvalue" id="parmvalue144721783518"><a name="parmvalue144721783518"></a><a name="parmvalue144721783518"></a>“4”</span>。取值范围：compress_ratio≥1。</p>
<p id="p1157915132617"><a name="p1157915132617"></a><a name="p1157915132617"></a>&lt;coarse_centroid_num&gt;：L1簇聚类中心个数，默认值为“16384”。</p>
<p id="p656513128471"><a name="p656513128471"></a><a name="p656513128471"></a>&lt;core_num&gt;：<span id="ph129021410310"><a name="ph129021410310"></a><a name="ph129021410310"></a>昇腾AI处理器</span>AI Core的个数，默认为<span class="parmvalue" id="parmvalue18901171585410"><a name="parmvalue18901171585410"></a><a name="parmvalue18901171585410"></a>“2”</span>。不指定该值时，根据&lt;npu_type&gt;配置：当npu_type配置为310时，&lt;core_num&gt;配置为2；当npu_type配置为310P时，&lt;core_num&gt;配置为8。</p>
<p id="p1489519612244"><a name="p1489519612244"></a><a name="p1489519612244"></a>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为<span class="parmvalue" id="parmvalue12408135012357"><a name="parmvalue12408135012357"></a><a name="parmvalue12408135012357"></a>“0”</span>，无需设置。</p>
<p id="p176142029102418"><a name="p176142029102418"></a><a name="p176142029102418"></a>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认值为<span class="parmvalue" id="parmvalue1756444519167"><a name="parmvalue1756444519167"></a><a name="parmvalue1756444519167"></a>“32”</span>。取值范围：1≤pool_size≤32。</p>
<p id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持<span id="ph97104582114"><a name="ph97104582114"></a><a name="ph97104582114"></a><term id="zh-cn_topic_0000001519959665_term169221139190"><a name="zh-cn_topic_0000001519959665_term169221139190"></a><a name="zh-cn_topic_0000001519959665_term169221139190"></a>Atlas 200/300/500 推理产品</term></span>以及<span id="ph19590185162111"><a name="ph19590185162111"></a><a name="ph19590185162111"></a>Atlas 推理系列产品</span>，取值范围分别为：310、310P，默认为<span class="parmvalue" id="parmvalue68401116171220"><a name="parmvalue68401116171220"></a><a name="parmvalue68401116171220"></a>“310”</span>。</p>
<p id="p581105217324"><a name="p581105217324"></a><a name="p581105217324"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row15956133317485"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p11956183311486"><a name="p11956183311486"></a><a name="p11956183311486"></a>说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p1648417714614"><a name="p1648417714614"></a><a name="p1648417714614"></a>执行此命令，用户可以得到一组算子模型文件。</p>
</td>
</tr>
<tr id="row1329410259210"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p132941725162113"><a name="p132941725162113"></a><a name="p132941725162113"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul423417236171"></a><a name="ul423417236171"></a><ul id="ul423417236171"><li>&lt;dim_in&gt; ∈ {256}</li><li>&lt;compress_ratio&gt; ∈ {2, 4, 8}</li><li>&lt;coarse_centroid_num&gt; ∈ {1024, 2048, 4096, 8192, 16384, 32768}</li><li>&lt;dim_in&gt;可以被&lt;compress_ratio&gt;整除。</li></ul>
</td>
</tr>
</tbody>
</table>

**涉及算法<a name="section1931762794815"></a>**

[AscendIndexIVFSQT](#近似检索)


### FlatAT<a name="ZH-CN_TOPIC_0000001506414881"></a>

> [!NOTE] 说明 
>当前FlatAT算子配合IVF类型的算子使用，用来加速IVF类型算子的add、train等过程，不支持直接调用FlatAT算子。当前的add/train加速功能通过IVF中AscendIndexIVFConfig.useKmeansPP进行指定，此时仅支持训练规模在7,000,000以下的训练。

<a name="table17415417319"></a>
<table><tbody><tr id="row124224153110"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p194211412319"><a name="p194211412319"></a><a name="p194211412319"></a>用法</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p1442184113118"><a name="p1442184113118"></a><a name="p1442184113118"></a>python3 flat_at_generate_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -c &lt;code_num&gt; -p &lt;process_id&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row11421741163119"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p1742941193110"><a name="p1742941193110"></a><a name="p1742941193110"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p2421641133115"><a name="p2421641133115"></a><a name="p2421641133115"></a>&lt;dim&gt;：输入特征向量维度，默认值为“64”。</p>
<p id="p44274114316"><a name="p44274114316"></a><a name="p44274114316"></a>&lt;code_num&gt;：与输入特征作对比的底库特征数，默认值为“8192”。</p>
<p id="p16163133719589"><a name="p16163133719589"></a><a name="p16163133719589"></a>&lt;core_num&gt;：<span id="ph129021410310"><a name="ph129021410310"></a><a name="ph129021410310"></a>昇腾AI处理器</span>AI Core的个数，默认为<span class="parmvalue" id="parmvalue645218305563"><a name="parmvalue645218305563"></a><a name="parmvalue645218305563"></a>“2”</span>。不指定该值时，根据&lt;npu_type&gt;配置：当npu_type配置为310时，&lt;core_num&gt;配置为2；当npu_type配置为310P时，&lt;core_num&gt;配置为8。</p>
<p id="p1222812464588"><a name="p1222812464588"></a><a name="p1222812464588"></a>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为<span class="parmvalue" id="parmvalue19166152815386"><a name="parmvalue19166152815386"></a><a name="parmvalue19166152815386"></a>“0”</span>，无需设置。</p>
<p id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持<span id="ph97104582114"><a name="ph97104582114"></a><a name="ph97104582114"></a><term id="zh-cn_topic_0000001519959665_term169221139190"><a name="zh-cn_topic_0000001519959665_term169221139190"></a><a name="zh-cn_topic_0000001519959665_term169221139190"></a>Atlas 200/300/500 推理产品</term></span>以及<span id="ph19590185162111"><a name="ph19590185162111"></a><a name="ph19590185162111"></a>Atlas 推理系列产品</span>，取值范围分别为：310、310P，默认为<span class="parmvalue" id="parmvalue68401116171220"><a name="parmvalue68401116171220"></a><a name="parmvalue68401116171220"></a>“310”</span>。</p>
<p id="p1989920599326"><a name="p1989920599326"></a><a name="p1989920599326"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row142104123115"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p124294113115"><a name="p124294113115"></a><a name="p124294113115"></a>说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p19612144442216"><a name="p19612144442216"></a><a name="p19612144442216"></a>执行此命令，用户可以得到一组算子模型文件。</p>
<p id="p4425415312"><a name="p4425415312"></a><a name="p4425415312"></a>FlatAT算子主要用于在IVF场景，减少train和add的耗时。</p>
</td>
</tr>
<tr id="row1828715702415"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p12287757152416"><a name="p12287757152416"></a><a name="p12287757152416"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul49294250179"></a><a name="ul49294250179"></a><ul id="ul49294250179"><li>dim ∈ {64, 128, 256}</li><li>code_num ∈ {1024, 2048, 4096, 8192, 16384, 32768}</li></ul>
</td>
</tr>
</tbody>
</table>

**涉及算法<a name="section019718356489"></a>**

-   [AscendIndexIVFSQ](#近似检索)
-   [AscendIndexIVFSQT](#近似检索)


### FlatInt8AT<a name="ZH-CN_TOPIC_0000001456694972"></a>

<a name="table17415417319"></a>
<table><tbody><tr id="row124224153110"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p194211412319"><a name="p194211412319"></a><a name="p194211412319"></a>用法</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p1442184113118"><a name="p1442184113118"></a><a name="p1442184113118"></a>python3 flat_at_int8_generate_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -c &lt;code_num&gt; -p &lt;process_id&gt; --soc-version &lt;soc_version&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row11421741163119"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p1742941193110"><a name="p1742941193110"></a><a name="p1742941193110"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p483612454218"><a name="p483612454218"></a><a name="p483612454218"></a>&lt;core_num&gt;：<span id="ph71911442141813"><a name="ph71911442141813"></a><a name="ph71911442141813"></a>昇腾AI处理器</span>AI Core的个数，默认为<span class="parmvalue" id="parmvalue11837165055814"><a name="parmvalue11837165055814"></a><a name="parmvalue11837165055814"></a>“8”</span>。</p>
<p id="p157945377424"><a name="p157945377424"></a><a name="p157945377424"></a>&lt;dim&gt;：输入特征向量维度，默认值为“256”。</p>
<p id="p44274114316"><a name="p44274114316"></a><a name="p44274114316"></a>&lt;code_num&gt;：与输入特征作对比的底库特征数，默认值为“16384”。</p>
<p id="p1222812464588"><a name="p1222812464588"></a><a name="p1222812464588"></a>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为<span class="parmvalue" id="parmvalue13397924123910"><a name="parmvalue13397924123910"></a><a name="parmvalue13397924123910"></a>“0”</span>，无需设置。</p>
<p id="p716454113415"><a name="p716454113415"></a><a name="p716454113415"></a>&lt;soc_version&gt;：<span id="ph129021410310"><a name="ph129021410310"></a><a name="ph129021410310"></a>昇腾AI处理器</span>的型号，默认为<span class="parmvalue" id="parmvalue198811757104210"><a name="parmvalue198811757104210"></a><a name="parmvalue198811757104210"></a>“Ascend310P3”</span>，无需设置。</p>
<p id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;：硬件形态，当前支持<span id="ph19590185162111"><a name="ph19590185162111"></a><a name="ph19590185162111"></a>Atlas 推理系列产品</span>，默认为<span class="parmvalue" id="parmvalue68401116171220"><a name="parmvalue68401116171220"></a><a name="parmvalue68401116171220"></a>“310P”</span>，无需设置。</p>
<p id="p418616673318"><a name="p418616673318"></a><a name="p418616673318"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row142104123115"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p124294113115"><a name="p124294113115"></a><a name="p124294113115"></a>说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p19612144442216"><a name="p19612144442216"></a><a name="p19612144442216"></a>执行此命令，用户可以得到一组算子模型文件。</p>
<p id="p137541630185615"><a name="p137541630185615"></a><a name="p137541630185615"></a>FlatInt8AT优化<span id="ph9726111217394"><a name="ph9726111217394"></a><a name="ph9726111217394"></a>Atlas 推理系列产品</span>使用场景下，IVFSQT中train、add与update的耗时。</p>
</td>
</tr>
<tr id="row1828715702415"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p12287757152416"><a name="p12287757152416"></a><a name="p12287757152416"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul49294250179"></a><a name="ul49294250179"></a><ul id="ul49294250179"><li>dim ∈ {256}</li><li>code_num ∈ {1024, 2048, 4096, 8192, 16384, 32768}</li><li>soc_version ∈ {Ascend310P3}</li></ul>
</td>
</tr>
</tbody>
</table>

**涉及算法<a name="section16686174317488"></a>**

[AscendIndexIVFSQT](#近似检索)


### AICPU<a name="ZH-CN_TOPIC_0000001506414793"></a>

<a name="table4331184817108"></a>
<table><tbody><tr id="row1433117485104"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p733211482108"><a name="p733211482108"></a><a name="p733211482108"></a>用法</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p1333211482107"><a name="p1333211482107"></a><a name="p1333211482107"></a>python3 aicpu_generate_model.py --cores &lt;core_num&gt; -p &lt;process_id&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row2033244801010"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p4332164821010"><a name="p4332164821010"></a><a name="p4332164821010"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p193321480106"><a name="p193321480106"></a><a name="p193321480106"></a>&lt;core_num&gt;：<span id="ph71911442141813"><a name="ph71911442141813"></a><a name="ph71911442141813"></a>昇腾AI处理器</span>AI Core的个数，默认为<span class="parmvalue" id="parmvalue4332204851012"><a name="parmvalue4332204851012"></a><a name="parmvalue4332204851012"></a>“2”</span>。（预留参数，暂不使用）</p>
<p id="p43321548181012"><a name="p43321548181012"></a><a name="p43321548181012"></a>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为<span class="parmvalue" id="parmvalue12408135012357"><a name="parmvalue12408135012357"></a><a name="parmvalue12408135012357"></a>“0”</span>，无需设置。</p>
<p id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持<span id="ph97104582114"><a name="ph97104582114"></a><a name="ph97104582114"></a><term id="zh-cn_topic_0000001519959665_term169221139190"><a name="zh-cn_topic_0000001519959665_term169221139190"></a><a name="zh-cn_topic_0000001519959665_term169221139190"></a>Atlas 200/300/500 推理产品</term></span>、<span id="ph19590185162111"><a name="ph19590185162111"></a><a name="ph19590185162111"></a>Atlas 推理系列产品</span>和<span id="ph996833614580"><a name="ph996833614580"></a><a name="ph996833614580"></a><term id="zh-cn_topic_0000001094307702_term99602034117"><a name="zh-cn_topic_0000001094307702_term99602034117"></a><a name="zh-cn_topic_0000001094307702_term99602034117"></a>Atlas A2 推理系列产品</term></span>，默认为<span class="parmvalue" id="parmvalue68401116171220"><a name="parmvalue68401116171220"></a><a name="parmvalue68401116171220"></a>“310”</span>。如果无法确定具体的npu_type，则在安装昇腾AI处理器的服务器执行<strong id="b87057513481"><a name="b87057513481"></a><a name="b87057513481"></a>npu-smi info</strong>命令进行查询，将查询到的“Name”最后一位数字删除，即是npu_type的取值。对于<span id="ph12325145818223"><a name="ph12325145818223"></a><a name="ph12325145818223"></a>Atlas 800I A3 超节点服务器</span>，可以通过<strong id="b10641459664"><a name="b10641459664"></a><a name="b10641459664"></a>npu-smi info -t board -i 0 -c 0</strong>命令进行查询，获取<strong id="b172691331852"><a name="b172691331852"></a><a name="b172691331852"></a>NPU Name</strong>信息，910_<strong id="b223011104513"><a name="b223011104513"></a><a name="b223011104513"></a>NPU Name</strong>即是npu_type的取值。</p>
<p id="p13676151710337"><a name="p13676151710337"></a><a name="p13676151710337"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row1333284817107"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p103329483101"><a name="p103329483101"></a><a name="p103329483101"></a>说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p1433264811011"><a name="p1433264811011"></a><a name="p1433264811011"></a>执行此命令，用户可以得到一组算子模型文件。</p>
<p id="p7405349165018"><a name="p7405349165018"></a><a name="p7405349165018"></a>AICPU算子模型文件只需生成一次，会全部生成所有算法的算子。</p>
</td>
</tr>
</tbody>
</table>

**涉及算法<a name="section156851751144816"></a>**

-   [AscendIndexInt8Flat](#全量检索)
-   [AscendIndexFlat](#全量检索)
-   [AscendIndexSQ](#全量检索)
-   [AscendIndexCluster](#全量检索)
-   [AscendIndexIVFSQ](#近似检索)
-   [AscendIndexBinaryFlat](#近似检索)
-   [AscendIndexTS](#属性过滤检索)
-   [AscendIndexIVFSQT](#近似检索)


### BinaryFlat<a name="ZH-CN_TOPIC_0000001506615001"></a>

<a name="table4331184817108"></a>
<table><tbody><tr id="row1433117485104"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p733211482108"><a name="p733211482108"></a><a name="p733211482108"></a>用法</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p1333211482107"><a name="p1333211482107"></a><a name="p1333211482107"></a>python3 binary_flat_generate_model.py -d &lt;dim&gt; -q &lt;query_type&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt;</p>
</td>
</tr>
<tr id="row2033244801010"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p4332164821010"><a name="p4332164821010"></a><a name="p4332164821010"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p193321480106"><a name="p193321480106"></a><a name="p193321480106"></a>&lt;dim&gt;：二值化特征向量维度，dim ∈ { 256， 512，1024 }，默认值为“512”。</p>
<p id="p1474218499117"><a name="p1474218499117"></a><a name="p1474218499117"></a>&lt;query_type&gt;：检索类型，默认为<span class="parmvalue" id="parmvalue12920913153"><a name="parmvalue12920913153"></a><a name="parmvalue12920913153"></a>“uint8”</span>，当AscendIndexBinaryFlat算法的<a href="./api/approximate_retrieval.md#search接口">search接口</a>进行性能提升时，需要设置为<span class="parmvalue" id="parmvalue10202131541419"><a name="parmvalue10202131541419"></a><a name="parmvalue10202131541419"></a>“float”</span>。</p>
<p id="p43321548181012"><a name="p43321548181012"></a><a name="p43321548181012"></a>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为<span class="parmvalue" id="parmvalue12408135012357"><a name="parmvalue12408135012357"></a><a name="parmvalue12408135012357"></a>“0”</span>，无需设置。</p>
<p id="p670916785418"><a name="p670916785418"></a><a name="p670916785418"></a>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认为16。</p>
<p id="p7767123203320"><a name="p7767123203320"></a><a name="p7767123203320"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row1333284817107"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p103329483101"><a name="p103329483101"></a><a name="p103329483101"></a>说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p1533224810102"><a name="p1533224810102"></a><a name="p1533224810102"></a>无。</p>
</td>
</tr>
</tbody>
</table>

**涉及算法<a name="section6613359134811"></a>**

-   [AscendIndexBinaryFlat](#近似检索)
-   [AscendIndexTS](#属性过滤检索)


### Mask<a name="ZH-CN_TOPIC_0000001461181500"></a>

<a name="table4331184817108"></a>
<table><tbody><tr id="row1433117485104"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p733211482108"><a name="p733211482108"></a><a name="p733211482108"></a>用法</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p1197352833411"><a name="p1197352833411"></a><a name="p1197352833411"></a>python3 mask_generate_model.py -token &lt;max_token_cnt&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row2033244801010"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p4332164821010"><a name="p4332164821010"></a><a name="p4332164821010"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p16675175613311"><a name="p16675175613311"></a><a name="p16675175613311"></a>&lt;max_token_cnt&gt;：算子生成token的最大值，默认为2500，建议设置范围为[1, 300000]。</p>
<p id="p1315743173515"><a name="p1315743173515"></a><a name="p1315743173515"></a>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为<span class="parmvalue" id="parmvalue12408135012357"><a name="parmvalue12408135012357"></a><a name="parmvalue12408135012357"></a>“0”</span>，无需设置。</p>
<p id="p10793175223414"><a name="p10793175223414"></a><a name="p10793175223414"></a>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认为16。</p>
<p id="p1719018389440"><a name="p1719018389440"></a><a name="p1719018389440"></a>&lt;npu_type&gt;：硬件形态，只支持<span class="parmvalue" id="parmvalue68401116171220"><a name="parmvalue68401116171220"></a><a name="parmvalue68401116171220"></a>“310P”</span>。</p>
<p id="p688819307338"><a name="p688819307338"></a><a name="p688819307338"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row1333284817107"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p103329483101"><a name="p103329483101"></a><a name="p103329483101"></a>说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p1896717588333"><a name="p1896717588333"></a><a name="p1896717588333"></a>无。</p>
</td>
</tr>
</tbody>
</table>

**涉及接口<a name="section1345318864915"></a>**

[AscendIndexTS](#属性过滤检索)


### IVFSP<a name="ZH-CN_TOPIC_0000001635696757"></a>

IVFSP检索当前只支持硬件形态“310P”，涉及以下几种类型的模型文件生成：

-   ivfsp\_generate\_model.py：IVFSP业务算子模型文件生成，具体请参见[IVFSP业务算子模型文件生成](#section11272703813)。
-   ivfsp\_aicpu\_generate\_model.py：IVFSP AICPU算子模型文件生成，具体请参见[IVFSP AICPU算子模型文件生成](#section10476137113814)。
-   ivfsp\_generate\_pyacl\_model.py：IVFSP训练码本时需要的训练算子模型文件生成，具体请参见[IVFSP训练算子模型文件生成](#section51314823813)。

**IVFSP业务算子模型文件生成<a ID="section11272703813"></a>**

<a name="table4331184817108"></a>
<table><tbody><tr id="row1433117485104"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p733211482108"><a name="p733211482108"></a><a name="p733211482108"></a>用法</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p828815402278"><a name="p828815402278"></a><a name="p828815402278"></a>python3 ivfsp_generate_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -nonzero_num &lt;low_dim&gt; -nlist &lt;k&gt; -handle_batch &lt;handle_batch&gt; -code_num &lt;code_num&gt; -p &lt;process_id&gt; --pool &lt;pool_size&gt;</p>
</td>
</tr>
<tr id="row2033244801010"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p4332164821010"><a name="p4332164821010"></a><a name="p4332164821010"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p472394917248"><a name="p472394917248"></a><a name="p472394917248"></a>&lt;core_num&gt;：AI Core的个数，默认值为“8”，无需设置。</p>
<p id="p19724949202417"><a name="p19724949202417"></a><a name="p19724949202417"></a>&lt;dim&gt;：特征向量维度，默认值为“256”。</p>
<p id="p472484911241"><a name="p472484911241"></a><a name="p472484911241"></a>&lt;low_dim&gt;：特征向量压缩后非零维度个数，默认值为“32”。</p>
<p id="p1572494913240"><a name="p1572494913240"></a><a name="p1572494913240"></a>&lt;k&gt;：簇聚类中心个数。与<a href="#section51314823813">IVFSP训练算子模型文件生成</a>中的&lt;k&gt;保持一致，默认值为“1024”。</p>
<p id="p1972464912414"><a name="p1972464912414"></a><a name="p1972464912414"></a>&lt;handle_batch&gt;：检索时每次下发计算的候选桶数量，默认值为“32”。</p>
<p id="p1272411499246"><a name="p1272411499246"></a><a name="p1272411499246"></a>&lt;code_num&gt;：检索时每次下发计算的每个桶的最大样本数量，若桶太大，程序会自动根据code_num将桶拆成多次算子下发计算距离。与<a href="#section51314823813">IVFSP训练算子模型文件生成</a>中的&lt;codebook_batch_size&gt;保持一致，默认值为“32768”。</p>
<p id="p5724134992412"><a name="p5724134992412"></a><a name="p5724134992412"></a>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为<span class="parmvalue" id="parmvalue12408135012357"><a name="parmvalue12408135012357"></a><a name="parmvalue12408135012357"></a>“0”</span>，无需设置。</p>
<p id="p1626301617420"><a name="p1626301617420"></a><a name="p1626301617420"></a>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认值为“16”。</p>
<p id="p11852192143213"><a name="p11852192143213"></a><a name="p11852192143213"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row1333284817107"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p103329483101"><a name="p103329483101"></a><a name="p103329483101"></a>说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p3370256123818"><a name="p3370256123818"></a><a name="p3370256123818"></a>执行此命令，用户可以得到一组用于IVFSP检索时的AI Core算子模型文件，用户需要自行修改命令中的参数。</p>
</td>
</tr>
<tr id="row1827720142259"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p327810141252"><a name="p327810141252"></a><a name="p327810141252"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul2098017146167"></a><a name="ul2098017146167"></a><ul id="ul2098017146167"><li>当dim ∈ {64, 128, 256}时，k∈ {256, 512, 1024, 2048, 4096, 8192, 16384}；当dim ∈ {512, 768}时，k∈ {256, 512, 1024, 2048}。</li><li>low_dim需为16的倍数且小于等于min(128, dim)。</li><li>handle_batch需为16的倍数，且16 ≤ handle_batch ≤ 240。</li><li>0 &lt; pool_size ≤ 32。</li></ul>
</td>
</tr>
</tbody>
</table>

**IVFSP AICPU算子模型文件生成<a id="section10476137113814"></a>**

<a name="table1844216303913"></a>
<table><tbody><tr id="row124438353916"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p1944314323914"><a name="p1944314323914"></a><a name="p1944314323914"></a>用法</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p164439315396"><a name="p164439315396"></a><a name="p164439315396"></a>python3 ivfsp_aicpu_generate_model.py --cores &lt;core_num&gt; -p &lt;process_id&gt;</p>
</td>
</tr>
<tr id="row1344373183918"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p244314363910"><a name="p244314363910"></a><a name="p244314363910"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p11745153613262"><a name="p11745153613262"></a><a name="p11745153613262"></a>&lt;core_num&gt;：AI Core的个数，默认值为“8”，无需设置。</p>
<p id="p13745113617269"><a name="p13745113617269"></a><a name="p13745113617269"></a>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为<span class="parmvalue" id="parmvalue1134717100258"><a name="parmvalue1134717100258"></a><a name="parmvalue1134717100258"></a>“0”</span>，无需设置。</p>
<p id="p31671221377"><a name="p31671221377"></a><a name="p31671221377"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row44439314393"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p1444316303912"><a name="p1444316303912"></a><a name="p1444316303912"></a>说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p144431434393"><a name="p144431434393"></a><a name="p144431434393"></a>执行此命令，用户可以得到一组用于IVFSP检索时的AICPU算子模型文件。</p>
</td>
</tr>
</tbody>
</table>

**IVFSP训练算子模型文件生成<a id="section51314823813"></a>**

<a name="table142311552394"></a>
<table><tbody><tr id="row12231105113915"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p62311050394"><a name="p62311050394"></a><a name="p62311050394"></a>用法</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p192321351392"><a name="p192321351392"></a><a name="p192321351392"></a>python3 ivfsp_generate_pyacl_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -nonzero_num &lt;low_dim&gt; -nlist &lt;k&gt; -batch_size &lt;batch_size&gt; -code_num &lt;codebook_batch_size&gt; -p &lt;process_id&gt;</p>
</td>
</tr>
<tr id="row723219523911"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p1023265163915"><a name="p1023265163915"></a><a name="p1023265163915"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p27921438102719"><a name="p27921438102719"></a><a name="p27921438102719"></a>&lt;core_num&gt;：AI Core的个数，默认值为“8”，无需设置。</p>
<p id="p19792438182711"><a name="p19792438182711"></a><a name="p19792438182711"></a>&lt;dim&gt;：特征向量维度，默认值为“256”。</p>
<p id="p0792163819273"><a name="p0792163819273"></a><a name="p0792163819273"></a>&lt;low_dim&gt;：特征向量压缩后非零维度个数，默认值为“32”。</p>
<p id="p20792133842718"><a name="p20792133842718"></a><a name="p20792133842718"></a>&lt;k&gt;：簇聚类中心个数。与<a href="#section11272703813">IVFSP业务算子模型文件生成</a>中的&lt;k&gt;保持一致，默认值为“1024”。</p>
<p id="p15792438122717"><a name="p15792438122717"></a><a name="p15792438122717"></a>&lt;batch_size&gt;：训练时以batch_size大小执行训练，默认值为“32768”。</p>
<p id="p87921838132719"><a name="p87921838132719"></a><a name="p87921838132719"></a>&lt;codebook_batch_size&gt;：训练时每次最大按codebook_batch_size样本数量操作码本，必须为2的幂次。与<a href="#section11272703813">IVFSP业务算子模型文件生成</a>中的&lt;code_num&gt;保持一致，默认值为“32768”。</p>
<p id="p17792838132718"><a name="p17792838132718"></a><a name="p17792838132718"></a>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为<span class="parmvalue" id="parmvalue11499918132520"><a name="parmvalue11499918132520"></a><a name="parmvalue11499918132520"></a>“0”</span>，无需设置。</p>
<p id="p194632813370"><a name="p194632813370"></a><a name="p194632813370"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row182322051393"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p1223218512395"><a name="p1223218512395"></a><a name="p1223218512395"></a>说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p12232650398"><a name="p12232650398"></a><a name="p12232650398"></a>执行此命令，用户可以得到一组用于IVFSP检索时的算子模型文件，用户需要自行修改命令中的参数。生成的IVFSP训练算子模型文件，保存在当前目录的子目录op_models_pyacl下。</p>
</td>
</tr>
<tr id="row1265606112615"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p10656267261"><a name="p10656267261"></a><a name="p10656267261"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul18345712132414"></a><a name="ul18345712132414"></a><ul id="ul18345712132414"><li>当dim ∈ {64, 128, 256}时，k∈ {256, 512, 1024, 2048, 4096, 8192, 16384}；当dim ∈ {512, 768}时，k∈ {256, 512, 1024, 2048}。</li><li>low_dim需为16的倍数且小于等于min(128, dim)。</li><li>batch_size需为16的倍数。</li><li>codebook_batch_size需为16的倍数。</li></ul>
</td>
</tr>
</tbody>
</table>


### VSTAR<a name="ZH-CN_TOPIC_0000002044867041"></a>

VSTAR检索当前只支持Atlas 推理系列产品，涉及VSTAR业务算子模型文件（vstar\_generate\_models.py）生成，具体请参见[VSTAR](#vstar)。

算子生成环境需要跟码本生成保持一致，具体请参见[总体说明](#总体说明)。

**VSTAR业务算子模型文件生成<a name="section11272703813"></a>**

<a name="table4331184817108"></a>
<table><tbody><tr id="row1433117485104"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p733211482108"><a name="p733211482108"></a><a name="p733211482108"></a>用法</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p828815402278"><a name="p828815402278"></a><a name="p828815402278"></a>python3 vstar_generate_models.py --dim &lt;dim&gt; --nlistL1 &lt;nlist1&gt;  --subDimL1 &lt;sub_dim1&gt;  --nProbeL1 &lt;nprobe1&gt; --nProbeL2 &lt;nprobe2&gt; --segmentNumL3 &lt;segment&gt; --pool &lt;pool_size&gt;</p>
</td>
</tr>
<tr id="row2033244801010"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p4332164821010"><a name="p4332164821010"></a><a name="p4332164821010"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p472394917248"><a name="p472394917248"></a><a name="p472394917248"></a>&lt;dim&gt;：特征向量维度，默认值为“256”。</p>
<p id="p1572494913240"><a name="p1572494913240"></a><a name="p1572494913240"></a>&lt;nlist1&gt;：一级簇聚类中心个数。默认值为“1024”。</p>
<p id="p1972464912414"><a name="p1972464912414"></a><a name="p1972464912414"></a>&lt;nprobe1&gt;：检索时每次下发计算时的一级候选桶数量，默认值为“[72]”。</p>
<p id="p75549202383"><a name="p75549202383"></a><a name="p75549202383"></a>&lt;nprobe2&gt;：检索时每次下发计算时的二级候选桶数量，默认值为“[64, 296]”。</p>
<p id="p193458505381"><a name="p193458505381"></a><a name="p193458505381"></a>&lt;sub_dim1&gt;：检索时一级降维后的维度大小，默认值为“32”。</p>
<p id="p686818358392"><a name="p686818358392"></a><a name="p686818358392"></a>&lt;segment&gt;：检索时从nprobe2中用于搜索数据段数，默认值“[512, 1000, 1504]”。</p>
<p id="p14215717122817"><a name="p14215717122817"></a><a name="p14215717122817"></a>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认“16”。</p>
<p id="p2216353369"><a name="p2216353369"></a><a name="p2216353369"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row1333284817107"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p103329483101"><a name="p103329483101"></a><a name="p103329483101"></a>说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p3370256123818"><a name="p3370256123818"></a><a name="p3370256123818"></a>执行此命令，用户可以得到一组用于VSTAR检索时的AI Core和AICPU算子模型文件，用户需要自行修改命令中的参数。</p>
</td>
</tr>
<tr id="row1827720142259"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p327810141252"><a name="p327810141252"></a><a name="p327810141252"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul144021537172510"></a><a name="ul144021537172510"></a><ul id="ul144021537172510"><li>dim ∈ {128, 256, 512, 1024}。</li><li>nlist1 ∈ {256, 512, 1024}。</li><li>sub_dim1 ∈ {32，64，128}。sub_dim1必须小于dim。</li><li>nprobe1 ∈ (16, nlist1]。nprobe1是int类型的列表，且列表中的数值必须是8的整数倍。</li><li>nprobe2 ∈ (16, nprobe1 * n]。当dim为1024时n为16，其余维度n为32，nprobe2是int类型的列表，且列表中的数值必须是8的整数倍。</li><li>segment ∈ (100, 5000]。segment是int类型的列表，且segment必须是8的整数倍。</li><li>pool_size∈[1, 32]。运行脚本前请先确定宿主机最大能支持的进程数量合理设置。</li></ul>
</td>
</tr>
</tbody>
</table>

**涉及算法<a name="section16686174317488"></a>**

[AscendIndexVStar](./api/approximate_retrieval.md#ascendindexvstar)

[AscendIndexGreat](./api/approximate_retrieval.md#ascendindexgreat)


### IVFFLAT<a name="ZH-CN_TOPIC_0000002478096638"></a>

<a name="table4331184817108"></a>
<table><tbody><tr id="row1433117485104"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.1.1"><p id="p733211482108"><a name="p733211482108"></a><a name="p733211482108"></a>用法</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.1.1 "><p id="p828815402278"><a name="p828815402278"></a><a name="p828815402278"></a>python3 ivfflat_generate_model.py -d &lt;dim&gt; -c &lt;coarse_centroid_num&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</p>
</td>
</tr>
<tr id="row2033244801010"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.2.1"><p id="p4332164821010"><a name="p4332164821010"></a><a name="p4332164821010"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.2.1 "><p id="p472394917248"><a name="p472394917248"></a><a name="p472394917248"></a>&lt;dim&gt;：特征向量维度，默认值为“128”。</p>
<p id="p1572494913240"><a name="p1572494913240"></a><a name="p1572494913240"></a>&lt;coarse_centroid_num&gt;：一级簇聚类中心个数。默认值为“<span id="ph1658923911236"><a name="ph1658923911236"></a><a name="ph1658923911236"></a>16384</span>”。</p>
<p id="p1149272010268"><a name="p1149272010268"></a><a name="p1149272010268"></a>&lt;core_num&gt;：<span id="ph129021410310"><a name="ph129021410310"></a><a name="ph129021410310"></a>昇腾AI处理器</span>AI Core的个数，默认为“40”。不指定该值时，根据&lt;npu_type&gt;配置：当&lt;npu_type&gt;配置为910B3时，&lt;core_num&gt;配置为40。</p>
<p id="p1849218206267"><a name="p1849218206267"></a><a name="p1849218206267"></a>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。</p>
<p id="p849220202265"><a name="p849220202265"></a><a name="p849220202265"></a>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认值为“10”。</p>
<p id="p174921720102611"><a name="p174921720102611"></a><a name="p174921720102611"></a>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持<span id="ph996833614580"><a name="ph996833614580"></a><a name="ph996833614580"></a><term id="zh-cn_topic_0000001094307702_term99602034117"><a name="zh-cn_topic_0000001094307702_term99602034117"></a><a name="zh-cn_topic_0000001094307702_term99602034117"></a>Atlas A2 推理系列产品</term></span>和<span id="ph167891051175217"><a name="ph167891051175217"></a><a name="ph167891051175217"></a><term id="zh-cn_topic_0000001094307702_term1977918118532"><a name="zh-cn_topic_0000001094307702_term1977918118532"></a><a name="zh-cn_topic_0000001094307702_term1977918118532"></a>Atlas A3 推理系列产品</term></span>，默认值为“910B4”。如果无法确定具体的npu_type，则在安装<span id="ph16510123015103"><a name="ph16510123015103"></a><a name="ph16510123015103"></a>昇腾AI处理器</span>的服务器执行<strong id="b1611533911102"><a name="b1611533911102"></a><a name="b1611533911102"></a>npu-smi info</strong>命令进行查询，将查询到的“Name”最后一位数字删除，即是npu_type的取值。对于<span id="ph1411710191414"><a name="ph1411710191414"></a><a name="ph1411710191414"></a>Atlas 800I A3 超节点服务器</span>，可以通过<strong id="b12401152416117"><a name="b12401152416117"></a><a name="b12401152416117"></a>npu-smi info -t board -i 0 -c 0</strong>命令进行查询，获取NPU Name信息，910_NPU Name即是npu_type的取值。</p>
<p id="p2216353369"><a name="p2216353369"></a><a name="p2216353369"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row1333284817107"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.3.1"><p id="p103329483101"><a name="p103329483101"></a><a name="p103329483101"></a>说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.3.1 "><p id="p3370256123818"><a name="p3370256123818"></a><a name="p3370256123818"></a>执行此命令，用户可以得到一组算子模型文件，用户需要自行修改命令中的参数。</p>
</td>
</tr>
<tr id="row1827720142259"><th class="firstcol" valign="top" width="14.580000000000002%" id="mcps1.1.3.4.1"><p id="p327810141252"><a name="p327810141252"></a><a name="p327810141252"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="85.42%" headers="mcps1.1.3.4.1 "><a name="ul144021537172510"></a><a name="ul144021537172510"></a><ul id="ul144021537172510"><li>dim ∈ {64, 128, 256, 384, 512}。</li><li>&lt;coarse_centroid_num&gt; ∈ {1024, 2048, 4096, 8192, 16384, 32768}</li><li>0 ≤ &lt;pool_size&gt; ≤ 32</li></ul>
</td>
</tr>
</tbody>
</table>

**涉及算法<a name="section16686174317488"></a>**

[AscendIndexIVFFlat](./api/approximate_retrieval.md#ascendindexivfflat)


### VSTAR生成码本文件<a name="ZH-CN_TOPIC_0000002008789068"></a>

#### 总体说明<a name="ZH-CN_TOPIC_0000002045184529"></a>

**环境配置<a name="section12757124191817"></a>**

环境依赖库参见如下：

-   nnae（8.0.0 <= version  <  8.5.0）
-   python（version \>= 3.9）
-   torch（version \>= 2.0.1）
-   torch\_npu（version \>= 2.0.1.post4）

-   numpy（version \>= 1.26.4）
-   scikit-learn（version \>= 1.4.1.post1）
-   tqdm（version ≥ 4.66.1）

torch、torch\_npu、numpy、scikit-learn和tqdm可通过**pip install**命令安装，执行命令参考如下。

```
pip install numpy tqdm scikit-learn torch_npu torch
```

CANN 8.5.0之前版本需要单独安装nnae。具体安装步骤如下：

1.  下载[nnae](https://www.hiascend.com/developer/download/community/result?module=cann&product=2&model=17)软件包。
2.  执行如下命令，增加可执行权限。

    ```
    chmod u+x ./Ascend-cann-nnae_{version}_linux-{arch}.run
    ```

3.  执行如下命令，进行安装。

    ```
    ./Ascend-cann-nnae_{version}_linux-{arch}.run --install
    ```

4.  按照安装提示信息设置环境变量。

    ```
    source /{nnae_installation_path}/nnae/set_env.sh
    ```

**注意事项<a name="section15462185871819"></a>**

-   若import torch，torch\_npu遇到下面的错误：

    ```
    .../libgomp.so: cannot allocate memory in static TLS block
    ```

    请执行export LD\_PRELOAD=.../libgomp.so（报错中出现的libgomp.so路径）

-   若安装numpy出现pip无法安装如下依赖时：

    ```
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

    请执行以下命令。

    ```
    pip install attrs cloudpickle decorator jinja2 ml-dtypes psutil scipy tornado absl-py
    ```

-   若训练码本遇到以下问题：

    ```
    OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata.
    Segmentation fault (core dumped)
    ```

    请执行：

    ```
    export OPENBLAS_NUM_THREADS=1
    ```

    该环境变量可能影响性能，码本训练完成后，建议设置回预设值。

-   --useOfflineCompile选项详细说明：

    在线算子编译耗时相比离线算子编译耗时较长。--useOfflineCompile选项用于控制是否使用离线算子编译，使用预先编译好的离线算子包执行。该方式需要用户提前安装单算子包。算子包安装指导如下：

    1.  下载[算子软件包](https://www.hiascend.com/developer/download/community/result?module=cann&product=2&model=17)。
    2.  执行如下命令，增加可执行权限。
        -   CANN  8.5.0之前版本

            ```
            chmod u+x ./Ascend-cann-kernels-{chip_type}_{version}_linux-{arch}.run
            ```

        -   CANN 8.5.0及之后版本

            ```
            chmod u+x ./Ascend-cann-{chip_type}-ops_{version}_linux-{arch}.run
            ```

    3.  执行如下命令，进行安装。
        -   CANN 8.5.0之前版本

            ```
            ./Ascend-cann-kernels-{chip_type}_{version}_linux-{arch}.run --install
            ```

        -   CANN 8.5.0及之后版本

            ```
            ./Ascend-cann-{chip_type}-ops_{version}_linux-{arch}.run --install
            ```

    4.  按照安装提示信息设置环境变量。
        -   CANN 8.5.0之前版本

            ```
            source /{kernels_installation_path}/kernels/set_env.sh
            ```

        -   CANN 8.5.0及之后版本

            ```
            source /usr/local/Ascend/cann/set_env.sh
            ```


#### 码本训练脚本<a name="ZH-CN_TOPIC_0000002008865568"></a>

训练涉及“vstar\_train\_codebook.py”脚本（训练脚本位于安装目录下的“tools/train”文件夹中），注意Python版本为3.9。

<a name="table48723587152"></a>
<table><tbody><tr id="row4899125881510"><th class="firstcol" valign="top" width="13.62%" id="mcps1.1.3.1.1"><p id="p1089905812153"><a name="p1089905812153"></a><a name="p1089905812153"></a>命令参考</p>
</th>
<td class="cellrowborder" valign="top" width="86.38%" headers="mcps1.1.3.1.1 "><p id="p989945810152"><a name="p989945810152"></a><a name="p989945810152"></a>python3 vstar_train_codebook.py --dataPath &lt;data_path&gt; --dim &lt;dim&gt; --codebookPath &lt;codebook_output_dir&gt; --nlistL1 &lt;nlist1&gt; --subDimL1 &lt;sub_dim1&gt;  --device &lt;device&gt; --batchSize &lt;batch_size&gt; --sample &lt;sample&gt; --useOfflineCompile</p>
</td>
</tr>
<tr id="row13899195817158"><th class="firstcol" valign="top" width="13.62%" id="mcps1.1.3.2.1"><p id="p198995588156"><a name="p198995588156"></a><a name="p198995588156"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="86.38%" headers="mcps1.1.3.2.1 "><p id="p112782011172018"><a name="p112782011172018"></a><a name="p112782011172018"></a>&lt;data_path&gt;：需要训练码本的原始数据路径，需要保证数据真实存在。该参数为必填项。</p>
<p id="p4899195861514"><a name="p4899195861514"></a><a name="p4899195861514"></a>&lt;dim&gt;：特征向量维度。与VSTAR训练算子模型文件生成的&lt;dim&gt;保持一致，默认值为<span class="parmvalue" id="parmvalue9207155164317"><a name="parmvalue9207155164317"></a><a name="parmvalue9207155164317"></a>“256”</span>。</p>
<p id="p9584201083120"><a name="p9584201083120"></a><a name="p9584201083120"></a>&lt;codebook_output_dir&gt;：最终生成的码本文件所存储的路径，生成的码本文件输出到的目录，用户应该保证此目录存在，且程序的执行用户对此目录具有写权限。出于安全加固考虑，目录层级中不能含有软链接。</p>
<p id="p689915891520"><a name="p689915891520"></a><a name="p689915891520"></a>&lt;nlist1&gt;：一级簇聚类中心个数。与VSTAR训练算子模型文件生成的&lt;nlist1&gt;保持一致，默认值为“1024”。</p>
<p id="p193458505381"><a name="p193458505381"></a><a name="p193458505381"></a>&lt;sub_dim1&gt;：检索时一级降维后的维度大小，与VSTAR训练算子模型文件生成的&lt;sub_dim1&gt;保持一致，默认值为“32”。</p>
<p id="p68991758111519"><a name="p68991758111519"></a><a name="p68991758111519"></a>&lt;device&gt;：设备逻辑ID，在指定的Device上执行训练，默认值为“1”。</p>
<p id="p089915814155"><a name="p089915814155"></a><a name="p089915814155"></a>&lt;batch_size&gt;：训练时以batch_size大小执行训练，参数范围(0，10240]，默认值为<span class="parmvalue" id="parmvalue17103111818276"><a name="parmvalue17103111818276"></a><a name="parmvalue17103111818276"></a>“10240”</span>。</p>
<p id="p168991358141520"><a name="p168991358141520"></a><a name="p168991358141520"></a>&lt;sample&gt;：训练用原始样本的采样率，0 &lt; ratio ≤ 1.0，默认为<span class="parmvalue" id="parmvalue497014364412"><a name="parmvalue497014364412"></a><a name="parmvalue497014364412"></a>“1.0”</span>。</p>
<p id="p1581627174815"><a name="p1581627174815"></a><a name="p1581627174815"></a>--useOfflineCompile：控制是否选择依赖算子包，使用离线算子编译，以获得性能提升。默认不开启。若开启，在命令行结尾增加该选项即可。详细说明请参见:VSTAR生成码本文件-总体说明- --useOfflineCompile选项详细说明。</p>
<p id="p575632015367"><a name="p575632015367"></a><a name="p575632015367"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row789917582151"><th class="firstcol" valign="top" width="13.62%" id="mcps1.1.3.3.1"><p id="p789955811157"><a name="p789955811157"></a><a name="p789955811157"></a>使用说明</p>
</th>
<td class="cellrowborder" valign="top" width="86.38%" headers="mcps1.1.3.3.1 "><a name="ul18156187102113"></a><a name="ul18156187102113"></a><ul id="ul18156187102113"><li>&lt;data_path&gt;原始数据大小需≤一千万1024维数据，即10,000,000 * 1024 * 4 = 40,960,000,000。</li><li>执行此命令，在&lt;codebook_output_dir&gt;对应的目录下生成新目录codebook_&lt;dim&gt;_&lt;nlist1&gt;_&lt;sub_dim1&gt;.bin，即为AscendIndexVStar和AscendIndexGreat所需使用到的码本文件。</li><li>当码本文件存在时，将执行覆盖写，此种情况程序执行用户应该是文件的属主。</li><li>在执行训练生成码本前，请先参考VSTAR，生成训练算子模型文件。</li></ul>
</td>
</tr>
</tbody>
</table>



### （可选）Python方式生成码本文件<a name="ZH-CN_TOPIC_0000001649848464"></a>

#### IVFSP训练脚本<a name="ZH-CN_TOPIC_0000001585736180"></a>

##### 环境配置<a name="ZH-CN_TOPIC_0000001585896272"></a>

环境依赖库参见如下：

-   numpy（version \> 1.16.0）
-   tqdm（version ≥ 4.65.0）
-   faiss-cpu（version = 1.10.0）

可通过**pip install**命令安装，命令执行参考如下。

```
pip install numpy tqdm faiss-cpu==1.10.0
```

执行训练脚本前，先执行如下命令设置环境变量。

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```


##### 训练脚本执行<a name="ZH-CN_TOPIC_0000001586056212"></a>

Index SDK提供两种训练脚本方式：

-   使用IVFSP算法的[trainCodeBook接口](./api/approximate_retrieval.md#traincodebook接口)进行训练（推荐使用该方式）。
-   使用“ivfsp\_train\_codebook.py”脚本进行训练。训练脚本位于安装目录下的“tools/train”文件夹中，注意Python版本为3.9.11。为了用户执行方便，提供了“ivfsp\_train\_codebook\_example.sh”样例脚本（脚本位于安装目录下的“tools/train”文件夹中），用户可在此文件上根据实际场景修改参数值，然后执行此脚本生成码本文件。

<a name="table48723587152"></a>
<table><tbody><tr id="row4899125881510"><th class="firstcol" valign="top" width="13.63%" id="mcps1.1.3.1.1"><p id="p1089905812153"><a name="p1089905812153"></a><a name="p1089905812153"></a>命令参考</p>
</th>
<td class="cellrowborder" valign="top" width="86.37%" headers="mcps1.1.3.1.1 "><p id="p989945810152"><a name="p989945810152"></a><a name="p989945810152"></a>python3 ivfsp_train_codebook.py --dim &lt;dim&gt; --nonzero_num &lt;nonzero_num&gt; --nlist &lt;nlist&gt; --num_iter &lt;num_iter&gt; --device &lt;device&gt; --batch_size &lt;batch_size&gt; --code_num &lt;code_num&gt; --ratio &lt;ratio&gt; --learn_data_path &lt;learn_data_path&gt; --codebook_output_dir &lt;codebook_output_dir&gt; --train_model_dir &lt;train_model_dir&gt;</p>
</td>
</tr>
<tr id="row13899195817158"><th class="firstcol" valign="top" width="13.63%" id="mcps1.1.3.2.1"><p id="p198995588156"><a name="p198995588156"></a><a name="p198995588156"></a>参数名称</p>
</th>
<td class="cellrowborder" valign="top" width="86.37%" headers="mcps1.1.3.2.1 "><p id="p4899195861514"><a name="p4899195861514"></a><a name="p4899195861514"></a>&lt;dim&gt;：特征向量维度。与IVFSP训练算子模型文件生成的&lt;dim&gt;保持一致，要求大于0。</p>
<p id="p1226910612333"><a name="p1226910612333"></a><a name="p1226910612333"></a>&lt;nonzero_num&gt;：特征向量压缩后非零维度个数，与IVFSP训练算子模型文件生成的&lt;low_dim&gt;保持一致，要求大于0。</p>
<p id="p689915891520"><a name="p689915891520"></a><a name="p689915891520"></a>&lt;nlist&gt;：簇聚类中心个数。与IVFSP训练算子模型文件生成的&lt;k&gt;保持一致，要求大于0。</p>
<p id="p089935861513"><a name="p089935861513"></a><a name="p089935861513"></a>&lt;num_iter&gt;：训练迭代次数参数，默认为20。迭代次数设置过大，会导致训练时长增加，要求大于0。</p>
<p id="p68991758111519"><a name="p68991758111519"></a><a name="p68991758111519"></a>&lt;device&gt;：设备逻辑ID，在指定的Device上执行训练，默认值为“0”。</p>
<p id="p089915814155"><a name="p089915814155"></a><a name="p089915814155"></a>&lt;batch_size&gt;：训练时以batch_size大小执行训练。与IVFSP训练算子模型文件生成的&lt;batch_size&gt;保持一致，要求大于0，小于等于32768，默认值为<span class="parmvalue" id="parmvalue17103111818276"><a name="parmvalue17103111818276"></a><a name="parmvalue17103111818276"></a>“32768”</span>。</p>
<p id="p9899195814150"><a name="p9899195814150"></a><a name="p9899195814150"></a>&lt;code_num&gt;：每次最大按code_num样本数量操作码本，必须为2的幂次。与IVFSP训练算子模型文件生成的&lt;codebook_batch_size&gt;保持一致，要求大于0，小于等于32768，默认值为<span class="parmvalue" id="parmvalue13486113643711"><a name="parmvalue13486113643711"></a><a name="parmvalue13486113643711"></a>“32768”</span>。</p>
<p id="p168991358141520"><a name="p168991358141520"></a><a name="p168991358141520"></a>&lt;ratio&gt;：训练用原始样本的采样率，0 &lt; ratio ≤ 1.0，默认为1.0。</p>
<p id="p1889985811157"><a name="p1889985811157"></a><a name="p1889985811157"></a>&lt;learn_data_path&gt;：训练用的原始特征文件路径，支持bin、npy格式，bin存储方式为行优先，数据类型为float32。</p>
<p id="p1089935810157"><a name="p1089935810157"></a><a name="p1089935810157"></a>&lt;codebook_output_dir&gt;：生成的码本文件输出到的目录，用户应该保证此目录存在，且程序的执行用户对此目录具有写权限；出于安全加固的考虑，此目录层级中不能含有软链接。</p>
<p id="p168997585150"><a name="p168997585150"></a><a name="p168997585150"></a>&lt;train_model_dir&gt;：IVFSP训练算子模型文件所在目录。</p>
<p id="p11852192143213"><a name="p11852192143213"></a><a name="p11852192143213"></a>--help | -h：查询帮助信息。</p>
</td>
</tr>
<tr id="row789917582151"><th class="firstcol" valign="top" width="13.63%" id="mcps1.1.3.3.1"><p id="p789955811157"><a name="p789955811157"></a><a name="p789955811157"></a>使用说明</p>
</th>
<td class="cellrowborder" valign="top" width="86.37%" headers="mcps1.1.3.3.1 "><a name="ul18156187102113"></a><a name="ul18156187102113"></a><ul id="ul18156187102113"><li>执行此命令，在&lt;codebook_output_dir&gt;对应的目录下生成文件codebook_&lt;dim&gt;_&lt;nonzero_num&gt;_&lt;nlist&gt;.bin和codebook_&lt;dim&gt;_&lt;nonzero_num&gt;_&lt;nlist&gt;.npy，codebook_&lt;dim&gt;_&lt;nonzero_num&gt;_&lt;nlist&gt;.bin即为AscendIndexIVFSP所需使用到的码本文件。</li><li>当码本文件存在时，将执行覆盖写，此种情况程序执行用户应该是文件的属主。</li><li>在执行训练生成码本前，请先参考IVFSP训练算子模型文件生成，生成训练算子模型文件。</li><li>learn_data_path指定的数据大小必须大于等于nonzero_num * nlist * sizeof(float32) 字节。</li></ul>
</td>
</tr>
</tbody>
</table>



#### 降维训练脚本<a name="ZH-CN_TOPIC_0000001681635905"></a>

**环境依赖<a name="section162431329141010"></a>**

-   安装Python3.9（支持Python3.9、Python3.10和Python3.11，推荐使用Python3.9）。
-   安装Faiss 1.10.0。可通过**pip install**命令安装，命令执行参考如下。

    ```
    pip install faiss-cpu==1.10.0
    ```

-   安装torch\_cpu和torch\_npu。安装方法参见[链接](https://gitee.com/ascend/pytorch)。请根据版本配套表，选择对应版本安装。

**训练模型<a name="section8422152014206"></a>**

本章节涉及的脚本的默认存放路径为：“tools/train/reduction”。

1.  训练模型。

    ```
    python3 call_train.py --dataset_dir=Dataset_Dir --val_dataset_dir=./valid --generate_val=True --save_path=./modelsDr --dim=512 --npu=0 --ratio=4 --metric=L2 --mode=train --train_size=100000 --epochs=20 --train_batch_size=8192 --infer_batch_size=128 --learning_rate=0.0005 --log_stride=500 --construct_neighbors=100 --queries_validation=1000
    ```

    |参数|说明|
    |--|--|
    |dataset_dir|数据集路径，类型为string，必须设置。目前实现默认读取base.npy，query.npy和gt.npy。若数据集为其他名称，可以自行实现数据集读取，并对该脚本get_train_data所在行做对应修改。例如。原代码为：<br>```# load dataset demo before training, modify here if you want to load your own dataset        #####################################################################        learn, base = get_train_data(args.dataset_dir, args.train_size)        #####################################################################```    <br>可修改为：<br>```# load dataset demo before training, modify here if you want to load your own dataset        #####################################################################        # learn, base = get_train_data(args.dataset_dir, args.train_size)        learn = np.fromfile(YOUR_LEARN_DATASET_DIR, dtype=np.float32).reshape((-1, YOUR_DATA_DIM))        base = np.fromfile(YOUR_BASE_DATASET_DIR, dtype=np.float32).reshape((-1, YOUR_DATA_DIM))        #####################################################################```|
    |val_dataset_dir|generate_val为True时有效，生成验证集的存放路径，类型为string，默认值为./validation/。|
    |generate_val|是否生成验证集。首次训练请设置为True。类型为bool，默认为False。|
    |save_path|模型存放路径。类型为string，必须设置。|
    |dim|可选，数据集维度。取值范围：[96, 128, 200, 256, 512, 2048]。类型为int，默认值为512。|
    |npu|训练所用的DeviceId，即设备号。类型为int。仅支持单卡训练，默认为CPU训练。|
    |ratio|可选，降维比例。取值范围：[2, 4, 8, 16]。类型为int，默认值为8。|
    |metric|训练模型时的距离度量标准，可选L2或IP。类型为string，默认值为L2。|
    |mode|可选，范围为[“train”,“infer”,“test”]，但当前仅支持“train”，默认为“train”，无需修改。|
    |train_size|训练集大小，取值范围小于整个数据集样本个数。用于读取数据集时随机采样部分数据进行训练。类型为int。若自行实现数据集读取，请根据train_size进行采样以防止训练速度过慢。默认值为100000，修改时要求该值大于0。|
    |epochs|训练迭代轮数。类型为int。迭代次数设置过大，会显著增加训练时长。默认为30，修改时要求该值大于0。|
    |train_batch_size|训练时的batch大小，默认为“8192”，类型为int。修改时要求该值大于0。|
    |infer_batch_size|推理时的batch大小，默认为“128”。类型为int。修改时要求该值大于0。|
    |learning_rate|学习率大小，默认为“0.0005”。类型为float。修改时要求该值大于0。|
    |log_stride|训练日志打印间隔（step），默认为“500”。类型为int。修改时要求该值大于0。|
    |construct_neighbors|构造训练集时所取的近邻的范围，用于构造降维所需的特殊训练集结构，默认为“100”。应根据数据集中每个人所对应的人脸数修改。类型为int。修改时要求该值大于0。|
    |queries_validation|构造验证集时所需查询向量的数量，类型为int。默认为“1000”，修改时要求该值大于0。|
    |--help \| -h|查询帮助信息。|


2.  生成OM模型。

    执行训练脚本前，先执行如下命令设置环境变量（根据CANN软件包的实际安装路径修改）。

    ```
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
    ```

    1.  生成精度为32的om模型。

        ```
        bash atc.sh {save_path} {om_name} {input_shape}
        ```

    2.  生成精度为16的om模型

        ```
        bash atc_16.sh {save_path} {om_name} {input_shape}
        ```

    -   \{save\_path\}：必选，表示模型存储的路径。路径中文件名需要以".onnx"或".pb"结尾，否则脚本会获取环境变量"framework"、"input\_format"等值，导致脚本执行异常。
    -   \{om\_name\}：可选，表示生成OM模型的名字，默认与onnx模型名字相同。
    -   \{input\_shape\}：可选，默认为onnx模型的输入维度，格式为actual\_input\_1:infer\_batch\_size,dim，建议使用默认值，不建议修改。
    -   **bash atc.sh**和**bash atc\_16.sh**仅支持Atlas 推理系列产品。

