# 使用指导

## 生成算子<a name="ZH-CN_TOPIC_0000001985832236"></a>

安装Index SDK后，需要依照本章节的指导，设置算子相关的环境变量，并生成算法所需要的算子。

> [!NOTE]
>
>- AscendIndexFlat算法L2和IP距离支持在线算子转换，如果环境变量**MX\_INDEX\_USE\_ONLINEOP**设置为1（设置命令：export MX\_INDEX\_USE\_ONLINEOP=1），则会在线转换算子并调用，不需要按照本章节生成离线算子。使用在线算子需要用户在应用程序的最后显式调用<b>\(void\)aclFinalize\(\)</b>（需要包含头文件：\#include "acl/acl.h"）。
>- 对于不支持在线算子的算法，如果设置了环境变量**MX\_INDEX\_USE\_ONLINEOP=1**，会导致程序运行失败。

**操作步骤<a name="section13749124217108"></a>**

1. 进入安装目录“mxIndex-_\{version\}_”，目录及文件名称如[表 Index SDK目录及文件名介绍](#table81133951612)所示。

    ```bash
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
    |modelpath|算子om文件存放目录。编译好算子之后，可将om文件放置于此文件夹。（可选）|
    |ops|包含custom_opp_\<arch>.run脚本，用于检索算法算子安装。|
    |script|包含卸载脚本uninstall.sh，用于卸载Index SDK安装包。|
    |tools|包含用于算子生成python脚本。|
    |version.info|包含版本相关信息。|

2. 进入“ops”目录，编译算子前需要设置“ASCEND\_HOME”、“ASCEND\_VERSION”和“ASCEND\_OPP\_PATH”环境变量，默认分别为\~/Ascend、\~/ascend-toolkit/latest和\~/Ascend/ascend-toolkit/latest/opp。

    ```bash
    export ASCEND_HOME=~/Ascend
    export ASCEND_VERSION=~/Ascend/ascend-toolkit/latest
    export ASCEND_OPP_PATH=~/Ascend/ascend-toolkit/latest/opp
    ```

    - “ASCEND\_HOME”表示CANN-toolkit软件安装后文件存储路径。
    - “ASCEND\_VERSION”表示当前使用的Ascend版本，如果ATC工具安装路径是“/usr/local/Ascend/ascend-toolkit/latest”则无需设置“ASCEND\_HOME”和“ASCEND\_VERSION”。
    - “ASCEND\_OPP\_PATH”表示算子库根目录，用户需要该目录的写权限。

    > [!NOTE]
    >“MAX\_COMPILE\_CORE\_NUMBER”环境变量用于指定图编译时可用的CPU核数，在算子运行时使用，当前默认为“1”，用户无需设置。

3. 根据实际系统架构执行对应脚本。

    - ARM架构：

        ```bash
        ./custom_opp_aarch64.run
        ```

    - x86\_64架构：

        ```bash
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

    > [!NOTE]
    >以下参数未展示在--help参数中，用户请勿直接使用。
    >- --xwin：使用xwin模式运行。
    >- --phase2：要求执行第二步动作。

4. 进入“tools”目录，生成所需算子。生成算子之前，需要先确认已经安装CANN的相关依赖。
    - 只生成使用的算法所需要的算子：先参考[算法介绍](#算法介绍)章节，确认算法所需要生成的算子后，再参考[自定义算子介绍](#自定义算子介绍)章节，生成对应的算子。
    - 批量生成所有算法的算子，方法如[表 批量生成算子](#table03891576018)所示。

        **表 3**  批量生成算子<a id="table03891576018"></a>

        <table><tbody>
        <tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong>python3 run_generate_model.py</strong> -m &lt;mode&gt; -t &lt;npu_type&gt; -p &lt;pipeline&gt; -pool &lt;pool_size&gt;</td></tr>
        <tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;mode&gt;：算法模式，&lt;mode&gt;支持ALL以及Flat，SQ8，IVFSQ8，INT8中的一种或多种，多种之间用逗号隔开，如：<strong>python3 run_generate_model.py</strong> <strong>-m Flat,IVFSQ8</strong>。默认全选，可以直接执行<strong>python3 run_generate_model.py</strong>。<br>&lt;npu_type&gt;：*npu_type*表示芯片名称。<br>● 对于Atlas 推理系列产品，可在安装昇腾AI处理器的服务器执行<strong>npu-smi info</strong>命令进行查询，将查询到的“Name”最后一位数字删除，即是npu_type的取值。<br>● 对于Atlas 800I A2 推理服务器，可在安装昇腾AI处理器的服务器执行<strong>npu-smi info</strong>命令进行查询，查询到的“Name”即是npu_type的取值。<br>● 对于Atlas 800I A3 超节点服务器，可以通过<strong>npu-smi info -t board -i 0 -c 0</strong>命令进行查询，获取<strong>NPU Name</strong>信息，910_<strong>NPU Name</strong>即是npu_type的取值。<br>&lt;pipeline&gt;：是否使用多线程并行流水生成算子模型，默认为true。设置为true时，使用默认的pool_size的值为32。<br>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小。<br>--help | -h：查询帮助信息。</td></tr>
        <tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">● 执行此命令，用户可以得到多组算子模型文件。执行命令前，用户需要更改当前目录下的para_table.xml文件，将所需的参数填入表中。<br>● 1 ≤ pool_size ≤ 32</td></tr>
        </tbody></table>

        > [!NOTE]
        >算子生成说明表格中的约束说明，代表业务中经常涉及的参数组合，使用其他参数运行异常请参见《[CANN ATC离线模型编译工具用户指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/devaids/atctool/atlasatc_16_0001.html)》。

5. 准备算子模型文件。

    - 可以将算子模型文件目录配置为环境变量“MX\_INDEX\_MODELPATH”（环境变量支持以\~开头的路径、相对路径和绝对路径，**路径中不能包含软链接**；使用该变量时将统一转化为绝对路径）。

        ```bash
        mv op_models/* $PWD/../modelpath
        export MX_INDEX_MODELPATH=`realpath $PWD/../modelpath`
        ```

    - 如未使用环境变量进行配置，需将算子模型文件移动到当前目录的“modelpath”目录下。

    算子生成后，请妥善保管相关om文件并确保文件不被篡改。

    > [!NOTE]
    >生成算子时如果出现报错：Failed to import Python module，请参照[NumPy的数据类型np.float\_ 已被移除](./07_faq.md#numpy的数据类型npfloat_-已被移除)解决。

## 算法介绍<a name="ZH-CN_TOPIC_0000001649848468"></a>

> [!NOTE]
>标准态部署主要使用AI CPU，Ctrl CPU和AI CPU的最佳推荐配比如下。
>
>- 使用<term>Atlas 推理系列产品</term>，建议设置为1:7。
具体设置命令参考[npu-smi命令](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/260RC1/re/npu/npusmi_053.html)。

### 全量检索<a name="ZH-CN_TOPIC_0000001698088061"></a>

**全量检索算法介绍<a name="section46312418528"></a>**

全量检索（Brute-force Search）是指对底库中的所有向量逐一计算距离，返回与查询向量距离最近的TopK结果。全量检索不进行任何剪枝或近似处理，因此检索精度最高，但计算量与底库规模成正比，适用于对精度要求严格、底库规模适中的场景。

<table><tbody>
<tr><td align="center" valign="middle"><strong>算法（API参考）</strong></td><td align="center" valign="middle"><strong>算法使用场景</strong></td><td align="center" valign="middle"><strong>需要生成的算子</strong></td><td width="100" align="center" valign="middle"><strong>样例链接</strong></td></tr>
<tr><td valign="middle"><a href="./api/01_full_retrieval/06_AscendIndexInt8Flat.md#ascendindexint8flat">AscendIndexInt8Flat</a></td><td valign="middle">● 特征类型：int8<br>● 特征维度：64, 128, 256, 384, 512, 768, 1024<br>● 距离类型：L2和IP<br>● 计算精度：高<br>● Device内存占用：较低<br>● 适应场景：精度要求高的暴力检索场景</td><td valign="middle">● <a href="#int8flat">INT8Flat</a><br>● <a href="#aicpu">AICPU</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexInt8Flat.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/01_full_retrieval/08_AscendIndexFlat.md#ascendindexflat">AscendIndexFlat</a></td><td valign="middle">● 特征类型：FP32、FP16<br>● 特征维度：32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096<br>● 距离类型：L2和IP<br>● 计算精度：高<br>● Device内存占用：高<br>● 适应场景：精度要求高的暴力检索场景；IP距离推荐在dim &gt; 128的场景下使用。</td><td valign="middle">● <a href="#flat">Flat</a><br>● <a href="#aicpu">AICPU</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexFlat.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/01_full_retrieval/11_AscendIndexSQ.md#ascendindexsq">AscendIndexSQ</a></td><td valign="middle">● 特征类型：FP32<br>● 特征维度：64, 128, 256, 384, 512, 768<br>● 距离类型：L2和IP<br>● 计算精度：高<br>● Device内存占用：较低（已量化为int8）<br>● 适应场景：精度要求较高的暴力检索场景</td><td valign="middle">● <a href="#sq8">SQ8</a><br>● <a href="#aicpu">AICPU</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexSQ.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/01_full_retrieval/02_AscendIndexCluster.md#ascendindexcluster">AscendIndexCluster</a></td><td valign="middle">● 特征类型：FP32<br>● 特征维度：32, 64, 128, 256, 384, 512<br>● 距离类型：IP<br>● 计算精度：高<br>● Device内存占用：较高<br>● 适应场景：只计算距离的聚类场景<br>● 仅支持Atlas 推理系列产品</td><td valign="middle">● <a href="#flat">Flat</a><br>● <a href="#aicpu">AICPU</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexCluster.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/01_full_retrieval/13_IndexIL.md#indexil">IndexIL</a></td><td valign="middle">需要运行在Device上，安装部署复杂，暂不推荐使用</td><td valign="middle">● <a href="#flat">Flat</a></td><td width="100" align="center" valign="middle">参考<a href="./api/01_full_retrieval/14_IndexILFlat.md#indexilflat">IndexILFlat</a></td></tr>
<tr><td valign="middle"><a href="./api/01_full_retrieval/15_AscendIndexILFlat.md#ascendindexilflat">AscendIndexILFlat</a></td><td valign="middle">● 特征类型：FP16、FP32<br>● 特征维度：32, 64, 128, 256, 384, 512<br>● 距离类型：IP<br>● 计算精度：高<br>● Device内存占用：较高<br>● 适应场景：只计算距离的聚类场景<br>● 仅支持Atlas 推理系列产品</td><td valign="middle">● <a href="#flat">Flat</a><br>● <a href="#aicpu">AICPU</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/tree/master/IndexSDK">链接</a></td></tr>
</tbody></table>

### 近似检索<a name="ZH-CN_TOPIC_0000001698168797"></a>

**近似检索算法介绍<a name="section46312418528"></a>**

近似检索（Approximate Nearest Neighbor Search）通过聚类、量化、图索引等方式对底库进行预处理或压缩，检索时仅计算部分向量距离，以牺牲少量精度换取显著的性能提升和内存节省。适用于亿级大库容、对时延敏感、可接受一定精度损失的场景。

<table><tbody>
<tr><td align="center" valign="middle"><strong>算法（API参考）</strong></td><td align="center" valign="middle"><strong>算法使用场景</strong></td><td align="center" valign="middle"><strong>需要生成的算子</strong></td><td width="100" align="center" valign="middle"><strong>样例链接</strong></td></tr>
<tr><td valign="middle"><a href="./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#ascendindexivfsp">AscendIndexIVFSP</a></td><td valign="middle">● 特征类型：FP32<br>● 特征维度：64, 128, 256, 512, 768<br>● 距离类型：L2<br>● 计算精度：中<br>● Device内存占用：低（压缩特征）<br>● 适应场景：适用于亿级底库（大库容），对性能要求较高，对精度损失有容忍的近似检索场景。<br>● 仅支持Atlas 推理系列产品</td><td valign="middle">● IVFSP业务算子<br>● IVFSP AICPU算子<br>● IVFSP训练算子（仅在需要通过训练生成码本文件时才使用到）<br>请参见<a href="#ivfsp">IVFSP</a>。</td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFSP.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/02_approximate_retrieval/07_AscendIndexIVFSQ.md#ascendindexivfsq">AscendIndexIVFSQ</a></td><td valign="middle">● 特征类型：FP32<br>● 特征维度：64, 128, 256, 384, 512<br>● 距离类型：L2和IP<br>● 计算精度：中<br>● Device内存占用：较低（量化为int8）<br>● 适应场景：IVFSQ算法作为性能-精度调节器，适用于对精度损失有容忍，但是对性能要求比较高的场景。</td><td valign="middle">● <a href="#ivfsq8">IVFSQ8</a><br>● <a href="#aicpu">AICPU</a><br>● <a href="#flatat">FlatAT</a>（仅在参数useKmeansPP设置为true的时候需要生成FlatAT算子）</td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFSQ.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/02_approximate_retrieval/09_AscendIndexIVFSQT.md#ascendindexivfsqt">AscendIndexIVFSQT</a></td><td valign="middle">● 特征类型：FP32<br>● 特征维度：256<br>● 距离类型：IP<br>● 计算精度：中<br>● Device内存占用：低（量化和降维）<br>● 适应场景：AscendIndexIVFSQT包含降维算法的三级检索IVFSQ算法，适用于亿级底库（大库容），对性能要求较高，对精度损失有容忍的近似检索场景。</td><td valign="middle">● <a href="#ivfsqt">IVFSQT</a><br>● <a href="#flatat">FlatAT</a><br>● <a href="#aicpu">AICPU</a><br>● <a href="#flatint8at">FlatInt8AT</a>（在Atlas 推理系列产品上时需要生成）</td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFSQT.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/02_approximate_retrieval/01_AscendIndexBinaryFlat.md#ascendindexbinaryflat">AscendIndexBinaryFlat</a></td><td valign="middle">● 特征类型：uint8二值化特征<br>● 特征维度：256, 512, 1024<br>● 距离类型：Hamming和IP<br>● 计算精度：高<br>● Device内存占用：低<br>● 适应场景：AscendIndexBinaryFlat类继承自Faiss的IndexBinary类，用于二值化特征检索。对内存占用要求较低，性能要求较高的场景。<br>● 仅支持Atlas 推理系列产品</td><td valign="middle">● <a href="#binaryflat">BinaryFlat</a><br>● <a href="#aicpu">AICPU</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexBinaryFlat.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/02_approximate_retrieval/11_AscendIndexVStar.md#ascendindexvstar">AscendIndexVStar</a></td><td valign="middle">● 特征类型：FP32<br>● 特征维度：128, 256, 512, 1024<br>● 距离类型：L2<br>● 计算精度：中<br>● Device内存占用：低（压缩特征）<br>● 适应场景：适用于千万级底库（大库容），对性能要求较高，对精度损失有容忍的近似检索场景。<br>● 仅支持Atlas 推理系列产品</td><td valign="middle">● VStar业务算子<br>● VStar AICPU算子<br>● VStar训练算子（仅在需要通过训练生成码本文件时才使用到）<br>请参见<a href="#vstar">VSTAR</a>。</td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexVStar.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/02_approximate_retrieval/12_AscendIndexGreat.md#ascendindexgreat">AscendIndexGreat</a></td><td valign="middle">● 特征类型：FP32<br>● 特征维度：128, 256, 512, 1024<br>● 距离类型：L2<br>● 计算精度：中<br>● Device内存占用：低（压缩特征）<br>● 适应场景：适用于千万级底库（大库容），对性能要求较高，对精度损失有容忍的近似检索场景。<br>● 仅支持Atlas 推理系列产品。（当mode为AKMode时，才需要生成算子）</td><td valign="middle">● VStar业务算子<br>● VStar AICPU算子<br>● VStar训练算子（仅在需要通过训练生成码本文件时才使用到）<br>请参见<a href="#vstar">VSTAR</a>。</td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexGreat.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/02_approximate_retrieval/14_AscendIndexIVFFlat.md#ascendindexivfflat">AscendIndexIVFFlat</a></td><td valign="middle">● 特征类型：FP32<br>● 特征维度：128<br>● 距离类型：IP<br>● 计算精度：中<br>● Device内存占用：中<br>● 适应场景：适用于亿级底库（大库容），对性能要求较高，对精度损失有容忍的近似检索场景。<br>● 仅支持Atlas A2 推理系列产品, Atlas A3 推理系列产品和Ascend 950 系列产品</td><td valign="middle">● <a href="#aicpu">AICPU</a><br>● <a href="#ivfflat">IVFFLAT</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFFlat.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/02_approximate_retrieval/15_AscendIndexIVFPQ.md#ascendindexivfpq">AscendIndexIVFPQ</a></td><td valign="middle">● 特征类型：FP32<br>● 特征维度：128<br>● 距离类型：L2<br>● 计算精度：中（近似检索）<br>● Device内存占用：低（基于PQ编码压缩向量）<br>● 适应场景：适用于亿级底库（大库容），对吞吐和时延要求较高，可接受一定精度损失的近似检索场景。<br>● 仅支持Ascend 950 系列产品</td><td valign="middle">● <a href="#aicpu">AICPU</a><br>● <a href="#ivfpq">IVFPQ</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFPQ.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/02_approximate_retrieval/16_AscendIndexIVFRaBitQ.md#ascendindexivfrabitq">AscendIndexIVFRaBitQ</a></td><td valign="middle">● 特征类型：FP32<br>● 特征维度：128<br>● 距离类型：L2 &amp; IP<br>● 计算精度：中<br>● Device内存占用：低（压缩特征）<br>● 适应场景：适用于亿级底库（大库容），对性能要求较高，对精度损失有容忍的近似检索场景。<br>● 仅支持Atlas A2 推理系列产品, Atlas A3 推理系列产品 和Ascend 950 系列产品</td><td valign="middle">● <a href="#aicpu">AICPU</a><br>● <a href="#ivfrabitq">IVFRaBitQ</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexIVFRabitQ.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/02_approximate_retrieval/18_AscendIndexCagra.md#ascendindexcagra">AscendIndexCagra</a></td><td valign="middle">● 特征类型：FP32<br>● 特征维度：64, 128, 256, 512<br>● 距离类型：L2<br>● 计算精度：中<br>● Device内存占用：低（RabitQ量化压缩）<br>● 适应场景：基于图检索的近似最近邻搜索，适用于亿级底库（大库容），对性能要求较高，对精度损失有容忍的近似检索场景。<br>● 仅支持Ascend 950 系列产品</td><td valign="middle">● <a href="#cagra">Cagra</a></td><td width="100" align="center" valign="middle"><a href="../../examples/TestAscendIndexCagra.cpp">链接</a></td></tr>
</tbody></table>

### 属性过滤检索<a name="ZH-CN_TOPIC_0000001649689168"></a>

**属性过滤检索算法介绍<a name="section46312418528"></a>**

属性过滤检索是指在向量检索的基础上，结合业务属性（如时间、空间、附加属性、自定义属性等）进行过滤，仅对满足属性条件的向量执行距离计算和排序，实现时空联合检索。适用于需要同时满足相似性和属性约束的场景。

<table><tbody>
<tr><td align="center" valign="middle"><strong>算法（API参考）</strong></td><td align="center" valign="middle"><strong>算法使用场景</strong></td><td align="center" valign="middle"><strong>需要生成的算子</strong></td><td width="100" align="center" valign="middle"><strong>样例链接</strong></td></tr>
<tr><td valign="middle"><a href="./api/03_attribute_filtering-based_retrieval/01_AscendIndexTS.md#ascendindexts">AscendIndexTS</a></td><td valign="middle">● 特征类型：uint8二值化特征、int8、FP32（具体算法不同而不同）<br>● 特征维度：具体算法不同而不同<br>● 距离类型：Hamming、Cos、IP、L2<br>● 计算精度：较高<br>● Device内存占用：较高<br>● 适应场景：需要过滤属性的时空库场景<br>● Cos和IP支持Atlas 推理系列产品，Atlas A2 推理系列产品，Atlas A3 推理系列产品<br>● Hamming距离仅支持Atlas 推理系列产品</td><td valign="middle">● <a href="#mask">Mask</a><br>● <a href="#binaryflat">BinaryFlat</a><br>● <a href="#int8flat">Int8Flat</a><br>● <a href="#flat">Flat</a><br>● <a href="#aicpu">AICPU</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIndexTS.cpp">链接</a></td></tr>
</tbody></table>

### 多Index批量检索<a name="ZH-CN_TOPIC_0000001649848472"></a>

**多Index批量检索介绍<a name="section46312418528"></a>**

多Index批量检索允许在单个Device上同时管理多个Index实例，通过一次调用对多个Index执行检索，减少Host与Device之间的交互次数，提升多库并发检索的整体吞吐。

<table><tbody>
<tr><td align="center" valign="middle"><strong>接口（API参考）</strong></td><td align="center" valign="middle"><strong>接口使用场景</strong></td><td align="center" valign="middle"><strong>可以使用本接口的算法</strong></td><td width="100" align="center" valign="middle"><strong>样例链接</strong></td></tr>
<tr><td valign="middle"><a href="./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchfaissindex接口">Search</a></td><td valign="middle">单Device进行多个Index检索。</td><td valign="middle">● <a href="./api/01_full_retrieval/11_AscendIndexSQ.md#ascendindexsq">AscendIndexSQ</a><br>● <a href="./api/01_full_retrieval/08_AscendIndexFlat.md#ascendindexflat">AscendIndexFlat</a><br>● <a href="./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#ascendindexivfsp">AscendIndexIVFSP</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchascendindex接口">Search</a></td><td valign="middle">单Device进行多个AscendIndex检索。</td><td valign="middle">● <a href="./api/01_full_retrieval/11_AscendIndexSQ.md#ascendindexsq">AscendIndexSQ</a><br>● <a href="./api/01_full_retrieval/08_AscendIndexFlat.md#ascendindexflat">AscendIndexFlat</a><br>● <a href="./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#ascendindexivfsp">AscendIndexIVFSP</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchascendindexint8接口">Search</a></td><td valign="middle">单Device进行多个AscendIndexInt8检索。</td><td valign="middle">● <a href="./api/01_full_retrieval/06_AscendIndexInt8Flat.md#ascendindexint8flat">AscendIndexInt8Flat</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilterfaissindex单filter接口">SearchWithFilter</a></td><td valign="middle">单Device进行多个Index带属性过滤（单filter）检索。</td><td valign="middle">● <a href="./api/01_full_retrieval/11_AscendIndexSQ.md#ascendindexsq">AscendIndexSQ</a><br>● <a href="./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#ascendindexivfsp">AscendIndexIVFSP</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilterascendindex单filter接口">SearchWithFilter</a></td><td valign="middle">单Device进行多个AscendIndex带属性过滤（单filter）检索。</td><td valign="middle">● <a href="./api/01_full_retrieval/11_AscendIndexSQ.md#ascendindexsq">AscendIndexSQ</a><br>● <a href="./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#ascendindexivfsp">AscendIndexIVFSP</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilterfaissindex多filter接口">SearchWithFilter</a></td><td valign="middle">单Device进行多个Index带过滤属性（多filter）检索。</td><td valign="middle">● <a href="./api/01_full_retrieval/11_AscendIndexSQ.md#ascendindexsq">AscendIndexSQ</a><br>● <a href="./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#ascendindexivfsp">AscendIndexIVFSP</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilterascendindex多filter接口">SearchWithFilter</a></td><td valign="middle">单Device进行多个AscendIndex带过滤属性（多filter）检索。</td><td valign="middle">● <a href="./api/01_full_retrieval/11_AscendIndexSQ.md#ascendindexsq">AscendIndexSQ</a><br>● <a href="./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#ascendindexivfsp">AscendIndexIVFSP</a></td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendMultiSearch.cpp">链接</a></td></tr>
</tbody></table>

### 其他功能<a name="ZH-CN_TOPIC_0000001698088065"></a>

**算法介绍<a name="section46312418528"></a>**

<table><tbody>
<tr><td align="center" valign="middle"><strong>算法（API参考）</strong></td><td align="center" valign="middle"><strong>算法需求（性能、场景差异）</strong></td><td align="center" valign="middle"><strong>如何调用</strong></td><td width="100" align="center" valign="middle"><strong>样例链接</strong></td></tr>
<tr><td valign="middle"><a href="./api/05_more_functions/01_IReduction.md#ireduction">IReduction</a></td><td valign="middle">IReduction是特征检索组件中降维方法的统一接口，目前支持<strong>PCAR</strong>和<strong>NN</strong>两种降维算法。</td><td valign="middle">通过ReductionConfig初始化，调用CreateReduction创建降维对象，然后进行train和reduce。</td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIReduction.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/05_more_functions/02_AscendNNInference.md#ascendnninference">AscendNNInference</a></td><td valign="middle">通过神经网络进行推理。</td><td valign="middle">通过AscendNNInference创建NN降维对象，然后进行infer降维。</td><td width="100" align="center" valign="middle"><a href="https://gitcode.com/Ascend/mindsdk-referenceapps/blob/master/IndexSDK/TestAscendIReduction.cpp">链接</a></td></tr>
<tr><td valign="middle"><a href="./api/05_more_functions/04_AscendCloner.md#ascendcloner">AscendCloner</a></td><td valign="middle">Index SDK提供了将NPU上的检索Index资源拷贝到CPU侧Faiss的操作，拷贝过程发生在内存中，原始NPU的Index上加载的数据会被拷贝到CPU侧的内存中，方便用户在CPU上使用相同的底库执行检索。</td><td valign="middle">index_ascend_to_cpu将AscendIndex拷贝生成一个CPU上的Index，index_cpu_to_ascend将CPU上的Index拷贝生成一个AscendIndex。</td><td width="100" align="center" valign="middle">无</td></tr>
</tbody></table>

## 自定义算子介绍<a name="ZH-CN_TOPIC_0000001456854988"></a>

### 自定义算子简介<a name="ZH-CN_TOPIC_0000001456695000"></a>

特征检索方案使用TIK算子开发实现特征距离计算逻辑，包含以下的自定义算子。

- [Flat距离计算算子](#flat)：得到特征底库数据和待检索的特征向量之间的距离（L2/IP）。
- [SQ8距离计算算子](#sq8)：得到SQ量化的特征底库数据和待检索的未量化特征向量之间的距离（L2/IP）。
- [IVFSQ8算子](#ivfsq8)：得到IVFSQ8算法所需要的算子。
- [INT8Flat距离计算算子](#int8flat)：得到INT8量化的特征底库数据和待检索的INT8量化特征向量之间的距离（L2/COS）。
- [IVFSQT算子](#ivfsqt)：得到IVFSQT算法一二三级所需的距离算子。
- [FlatAT算子](#flatat)：主要用于在IVF场景，减少train和add的耗时，其中“code\_num”等于“nlist”。
- [FlatInt8AT算子](#flatint8at)：优化在<term>Atlas 推理系列产品</term>下IVFSQT中train、add与update的耗时。
- [AICPU算子](#aicpu)：调度昇腾AI处理器的CPU完成排序等计算，充分利用硬件性能。
- [BinaryFlat算子](#binaryflat)：得到二值化算法所需算子。
- [Mask算子](#mask)：得到时空库属性过滤算法所需的Mask算子。
- [IVFSP算子](#ivfsp)：得到IVFSP算法所需的业务算子、AICPU算子，以及训练生成IVFSP码本时所需的训练算子。
- [VStar算子](#vstar)：得到VStar算法所需的业务算子、AICPU算子。
- [IVFFLAT](#ivfflat)：得到IVFFLAT算法一级二级所需的距离算子。
- [IVFPQ算子](#ivfpq)：得到IVFPQ算法一级二级三级所需的距离算子。
- [IVFRaBitQ算子](#ivfrabitq): 得到IVFRaBitQ所需的算子。
- [Cagra算子](#cagra)：得到Cagra图检索算法所需的算子。

### 算子生成说明<a name="ZH-CN_TOPIC_0000001456695052"></a>

#### Flat<a name="ZH-CN_TOPIC_0000001506495813"></a>

<a name="table3955133174816"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 flat_generate_model.py -d &lt;dim&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;dim&gt;：特征向量维度D，默认值为“512”。<br>&lt;core_num&gt;：昇腾AI处理器AI Core的个数，默认值为“8”。无需设置。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认值为“10”。<br>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持Atlas 推理系列产品，Atlas A2 推理系列产品，Atlas A3 推理系列产品，默认值为&quot;310P&quot;。<br>● 对于Atlas 推理系列产品，可在安装昇腾AI处理器的服务器执行<strong><code>npu-smi info</code></strong>命令进行查询，将查询到的“Name”最后一位数字删除，即是npu_type的取值。<br>● 对于Atlas 800I A2 推理服务器，可在安装昇腾AI处理器的服务器执行<strong><code>npu-smi info</code></strong>命令进行查询，查询到的“Name”即是npu_type的取值。<br>● 对于Atlas 800I A3 超节点服务器，可以通过<strong><code>npu-smi info -t board -i 0 -c 0</code></strong>命令进行查询，获取<strong><code>NPU Name</code></strong>信息，910_<strong><code>NPU Name</code></strong>即是npu_type的取值。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组距离计算算子模型文件，用户需要自行修改命令中的参数。例如对于Atlas 推理系列产品 生成512维算子：python3 flat_generate_model.py -d 512 -t 310P</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dim ∈ {32, 64, 128, 256, 384, 512, 768, 1024, 1408, 1536, 2048, 3072, 3584, 4096}<br>● 1 ≤ pool_size ≤ 32</td></tr>
</tbody></table>

**涉及算法<a name="section1467921619472"></a>**

- [AscendIndexFlat](#全量检索)
- [AscendIndexCluster](#全量检索)
- [IndexIL](#全量检索)
- [AscendIndexTS](#属性过滤检索)
- [Search（单device进行多个Index检索）](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchfaissindex接口)
- [Search（单device进行多个AscendIndex检索）](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchascendindex接口)

#### SQ8<a name="ZH-CN_TOPIC_0000001506614921"></a>

> [!NOTE]
>INT8Flat和SQ8的区别主要在于：INT8由外部进行量化，Index的输入特征是INT8类型，SQ8由Index内部量化，Index的输入特征是Float32类型。

<a name="table3955133174816"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 sq8_generate_model.py -d &lt;dim&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;dim&gt;：特征向量维度D，默认值为“128”。<br>&lt;core_num&gt;：昇腾AI处理器AI Core的个数，默认为“8”。不指定该值时，根据&lt;npu_type&gt;配置：当npu_type配置为310P时，&lt;core_num&gt;配置为8。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认值为“10”。<br>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持Atlas 推理系列产品，取值为：310P，默认为“310P”。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组SQ8距离计算算子模型文件，用户需要自行修改命令中的参数。例如对于Atlas 推理系列产品 生成512维算子：python3 sq8_generate_model.py -d 512 -t 310P</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dim ∈ {64, 128, 256, 384, 512, 768}<br>● 1 ≤ pool_size ≤ 32</td></tr>
</tbody></table>

**涉及算法<a name="section6413836184719"></a>**

- [AscendIndexSQ](#全量检索)
- [Search（单device进行多个Index检索）](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchfaissindex接口)
- [Search（单device进行多个AscendIndex检索）](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchascendindex接口)
- [SearchWithFilter（FaissIndex单filter）](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilterfaissindex单filter接口)
- [SearchWithFilter（AscendIndex单filter）](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilterascendindex单filter接口)
- [SearchWithFilter（FaissIndex多filter）](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilterfaissindex多filter接口)
- [SearchWithFilter（AscendIndex多filter）](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchwithfilterascendindex多filter接口)

#### IVFSQ8<a name="ZH-CN_TOPIC_0000001506614889"></a>

<a name="table3955133174816"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 ivfsq8_generate_model.py -d &lt;dim&gt; -c &lt;coarse_centroid_num&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;dim&gt;：特征向量维度D，默认值为“128”。<br>&lt;coarse_centroid_num&gt;：L1簇聚类中心个数，默认值为“16384”。<br>&lt;core_num&gt;：昇腾AI处理器AI Core的个数，默认为“8”。不指定该值时，根据&lt;npu_type&gt;配置：当npu_type配置为310P时，&lt;core_num&gt;配置为8。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认值为“10”。<br>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持Atlas 推理系列产品，取值为：310P，默认为“310P”。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组算子模型文件，用户需要自行修改命令中的参数。例如对于Atlas 推理系列产品 生成512维,nlist为1024算子：python3 ivfsq8_generate_model.py -d 512 -c 1024 -t 310P</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dim ∈ {64, 128, 256, 384, 512}<br>● coarse centroid num ∈ {1024, 2048, 4096, 8192, 16384, 32768}<br>● 1 ≤ pool_size ≤ 32</td></tr>
</tbody></table>

**涉及算法<a name="section14565105918474"></a>**

[AscendIndexIVFSQ](#近似检索)

#### INT8Flat<a name="ZH-CN_TOPIC_0000001456695008"></a>

> [!NOTE]
>INT8Flat和SQ8的区别主要在于：INT8由外部进行量化，Index的输入特征是INT8类型，SQ8由Index内部量化，Index的输入特征是Float32类型。

<a name="table3955133174816"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 int8flat_generate_model.py -d &lt;dim&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt; -code &lt;code_num&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;dim&gt;：特征向量维度D，默认值为“512”。<br>&lt;core_num&gt;：昇腾AI处理器AI Core的个数，默认为“8”。无需设置。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认值为“10”。<br>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持Atlas A2 推理系列产品、Atlas A3 推理系列产品，默认值为“310P”。<br>● 对于Atlas 推理系列产品，可在安装昇腾AI处理器的服务器执行<strong><code>npu-smi info</code></strong>命令进行查询，将查询到的“Name”最后一位数字删除，即是npu_type的取值。<br>● 对于Atlas 800I A2 推理服务器，可在安装昇腾AI处理器的服务器执行<strong><code>npu-smi info</code></strong>命令进行查询，查询到的&quot;Name&quot;即是npu_type的取值。<br>&lt;code_num&gt;：算子调用时底库分块大小，默认值为“262144”，不设置时默认生成所有code_num值的算子。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组算子模型文件，用户需要自行修改命令中的参数。例如对于Atlas 推理系列产品 生成512维算子：python3 int8flat_generate_model.py -d 512 -t 310P</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dim ∈ {64, 128, 256, 384, 512, 768, 1024}<br>● 1 ≤ pool_size ≤ 32<br>● code_num ∈ {16384, 32768, 65536, 131072, 262144}</td></tr>
</tbody></table>

**涉及算法<a name="section3261111214818"></a>**

- [AscendIndexInt8Flat](#全量检索)
- [AscendIndexTS](#属性过滤检索)
- [Search（单device进行多个AscendIndexInt8检索）](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md#searchascendindexint8接口)

#### IVFSQT<a name="ZH-CN_TOPIC_0000001506414677"></a>

> [!NOTE]
>
>为了减少train和add的耗时，需要生成FlatAT算子。其中，Flat的<dim\>需与IVFSQT的<dim\_in\>相同，Flat的<code\_num\>与IVFSQT的<coarse\_centroid\_num\>一致。

<a name="table3955133174816"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 ivfsqt_generate_model.py --cores &lt;core_num&gt; -d &lt;dim_in&gt; -r &lt;compress_ratio&gt; -c &lt;coarse_centroid_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;dim_in&gt;：输入特征向量维度，默认值为“256”。<br>&lt;compress_ratio&gt;：输入与输出维度的比值，默认值为“4”。取值范围：compress_ratio≥1。<br>&lt;coarse_centroid_num&gt;：L1簇聚类中心个数，默认值为“16384”。<br>&lt;core_num&gt;：昇腾AI处理器AI Core的个数，默认为“8”。不指定该值时，根据&lt;npu_type&gt;配置：当npu_type配置为310P时，&lt;core_num&gt;配置为8。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认值为“32”。取值范围：1≤pool_size≤32。<br>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持Atlas 推理系列产品，取值为：310P，默认为“310P”。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组算子模型文件。例如对于Atlas 推理系列产品 生成输入256维，输出64维，nlist1024算子：python3 ivfsqt_generate_model.py -d 256 -r 4 -c 1024 -t 310P</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● &lt;dim_in&gt; ∈ {256}<br>● &lt;compress_ratio&gt; ∈ {2, 4, 8}<br>● &lt;coarse_centroid_num&gt; ∈ {1024, 2048, 4096, 8192, 16384, 32768}<br>● &lt;dim_in&gt;可以被&lt;compress_ratio&gt;整除。</td></tr>
</tbody></table>

**涉及算法<a name="section1931762794815"></a>**

[AscendIndexIVFSQT](#近似检索)

#### FlatAT<a name="ZH-CN_TOPIC_0000001506414881"></a>

> [!NOTE]
>当前FlatAT算子配合IVF类型的算子使用，用来加速IVF类型算子的add、train等过程，不支持直接调用FlatAT算子。当前的add/train加速功能通过IVF中AscendIndexIVFConfig.useKmeansPP进行指定，此时仅支持训练规模在7,000,000以下的训练。

<a name="table17415417319"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 flat_at_generate_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -c &lt;code_num&gt; -p &lt;process_id&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;dim&gt;：输入特征向量维度，默认值为“64”。<br>&lt;code_num&gt;：与输入特征作对比的底库特征数，默认值为“8192”。<br>&lt;core_num&gt;：昇腾AI处理器AI Core的个数，默认为“8”。不指定该值时，根据&lt;npu_type&gt;配置：当npu_type配置为310P时，&lt;core_num&gt;配置为8。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持Atlas 推理系列产品，取值为：310P，默认为“310P”。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组算子模型文件。例如对于Atlas 推理系列产品 生成256维，nlist1024算子： python3 flat_at_generate_model.py -d 256 -c 1024 -t 310P<br>FlatAT算子主要用于在IVF场景，减少train和add的耗时。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dim ∈ {64, 128, 256}<br>● code_num ∈ {1024, 2048, 4096, 8192, 16384, 32768}</td></tr>
</tbody></table>

**涉及算法<a name="section019718356489"></a>**

- [AscendIndexIVFSQ](#近似检索)
- [AscendIndexIVFSQT](#近似检索)

#### FlatInt8AT<a name="ZH-CN_TOPIC_0000001456694972"></a>

<a name="table17415417319"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 flat_at_int8_generate_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -c &lt;code_num&gt; -p &lt;process_id&gt; --soc-version &lt;soc_version&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;core_num&gt;：昇腾AI处理器AI Core的个数，默认为“8”。<br>&lt;dim&gt;：输入特征向量维度，默认值为“256”。<br>&lt;code_num&gt;：与输入特征作对比的底库特征数，默认值为“16384”。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>&lt;soc_version&gt;：昇腾AI处理器的型号，默认为“Ascend310P3”，无需设置。<br>&lt;npu_type&gt;：硬件形态，当前支持Atlas 推理系列产品，默认为“310P”，无需设置。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组算子模型文件。例如对于Atlas 推理系列产品 生成256维，nlist1024算子： python3 flat_at_int8_generate_model.py -d 256 -c 1024 -t 310P<br>FlatInt8AT优化Atlas 推理系列产品使用场景下，IVFSQT中train、add与update的耗时。</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dim ∈ {256}<br>● code_num ∈ {1024, 2048, 4096, 8192, 16384, 32768}<br>● soc_version ∈ {Ascend310P3}</td></tr>
</tbody></table>

**涉及算法<a name="section16686174317488"></a>**

[AscendIndexIVFSQT](#近似检索)

#### AICPU<a name="ZH-CN_TOPIC_0000001506414793"></a>

<a name="table4331184817108"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 aicpu_generate_model.py --cores &lt;core_num&gt; -p &lt;process_id&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;core_num&gt;：昇腾AI处理器AI Core的个数，默认为“2”。（预留参数，暂不使用）<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持Atlas 推理系列产品和Atlas A2 推理系列产品、Atlas A3 推理系列产品，默认为“310P”。如果无法确定具体的npu_type，则在安装昇腾AI处理器的服务器执行<strong><code>npu-smi info</code></strong>命令进行查询，将查询到的“Name”最后一位数字删除，即是npu_type的取值。对于Atlas 800I A3 超节点服务器，可以通过<strong><code>npu-smi info -t board -i 0 -c 0</code></strong>命令进行查询，获取<strong><code>NPU Name</code></strong>信息，910_<strong><code>NPU Name</code></strong>即是npu_type的取值。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组算子模型文件。例如对于Atlas 推理系列产品 生成aicpu算子：python3 aicpu_generate_model.py -t 310P<br>AICPU算子模型文件只需生成一次，会全部生成所有算法的算子。</td></tr>
</tbody></table>

**涉及算法<a name="section156851751144816"></a>**

- [AscendIndexInt8Flat](#全量检索)
- [AscendIndexFlat](#全量检索)
- [AscendIndexSQ](#全量检索)
- [AscendIndexCluster](#全量检索)
- [AscendIndexIVFSQ](#近似检索)
- [AscendIndexBinaryFlat](#近似检索)
- [AscendIndexTS](#属性过滤检索)
- [AscendIndexIVFSQT](#近似检索)
- [AscendIndexIVFFlat](#近似检索)
- [AscendIndexIVFPQ](#近似检索)
- [AscendIndexIVFRaBitQ](#近似检索)

#### BinaryFlat<a name="ZH-CN_TOPIC_0000001506615001"></a>

<a name="table4331184817108"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 binary_flat_generate_model.py -d &lt;dim&gt; -q &lt;query_type&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;dim&gt;：二值化特征向量维度，dim ∈ { 256， 512，1024 }，默认值为“512”。<br>&lt;query_type&gt;：检索类型，默认为“uint8”，当AscendIndexBinaryFlat算法的<a href="./api/02_approximate_retrieval/01_AscendIndexBinaryFlat.md#ZH-CN_TOPIC_0000001456375288">search接口</a>进行性能提升时，需要设置为“float”。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认为16。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">例如对于Atlas 推理系列产品 生成256维 uint8类型算子：python3 binary_flat_generate_model.py -d 256。</td></tr>
</tbody></table>

**涉及算法<a name="section6613359134811"></a>**

- [AscendIndexBinaryFlat](#近似检索)
- [AscendIndexTS](#属性过滤检索)

#### Mask<a name="ZH-CN_TOPIC_0000001461181500"></a>

<a name="table4331184817108"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 mask_generate_model.py -token &lt;max_token_cnt&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;max_token_cnt&gt;：算子生成token的最大值，默认为2500，建议设置范围为[1, 300000]。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认为16。<br>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持Atlas 推理系列产品，Atlas A2 推理系列产品、Atlas A3 推理系列产品，默认值为“310P”。 <br>● 对于Atlas 推理系列产品，可在安装昇腾AI处理器的服务器执行<strong><code>npu-smi info</code></strong>命令进行查询，将查询到的“Name”最后一位数字删除，即是npu_type的取值。<br>● 对于Atlas 800I A2 推理服务器，可在安装昇腾AI处理器的服务器执行<strong><code>npu-smi info</code></strong>命令进行查询，查询到的“Name”即是npu_type的取值。<br>● 对于Atlas 800I A3 超节点服务器，可以通过<strong><code>npu-smi info -t board -i 0 -c 0</code></strong>命令进行查询，获取<strong><code>NPU Name</code></strong>信息，910_<strong><code>NPU Name</code></strong>即是npu_type的取值。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">例如对于Atlas 推理系列产品 生成token数量为300000的算子：python3 mask_generate_model.py -token 300000 -t 310P。</td></tr>
</tbody></table>

**涉及接口<a name="section1345318864915"></a>**

[AscendIndexTS](#属性过滤检索)

#### IVFSP<a name="ZH-CN_TOPIC_0000001635696757"></a>

IVFSP检索当前支持硬件形态“910B4”，涉及以下几种类型的模型文件生成：

- ivfsp\_generate\_model.py：IVFSP业务算子模型文件生成，具体请参见[IVFSP业务算子模型文件生成](#section11272703813)。
- ivfsp\_aicpu\_generate\_model.py：IVFSP AICPU算子模型文件生成，具体请参见[IVFSP AICPU算子模型文件生成](#section10476137113814)。
- ivfsp\_generate\_pyacl\_model.py：IVFSP训练码本时需要的训练算子模型文件生成，具体请参见[IVFSP训练算子模型文件生成](#section51314823813)。

**IVFSP业务算子模型文件生成<a ID="section11272703813"></a>**

<a name="table4331184817108"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 ivfsp_generate_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -nonzero_num &lt;low_dim&gt; -nlist &lt;k&gt; -handle_batch &lt;handle_batch&gt; -code_num &lt;code_num&gt; -p &lt;process_id&gt; --pool &lt;pool_size&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;core_num&gt;：AI Core的个数，默认值为“8”，无需设置。<br>&lt;dim&gt;：特征向量维度，默认值为“256”。<br>&lt;low_dim&gt;：特征向量压缩后非零维度个数，默认值为“32”。<br>&lt;k&gt;：簇聚类中心个数。与<a href="#section51314823813">IVFSP训练算子模型文件生成</a>中的&lt;k&gt;保持一致，默认值为“1024”。<br>&lt;handle_batch&gt;：检索时每次下发计算的候选桶数量，默认值为“32”。<br>&lt;code_num&gt;：检索时每次下发计算的每个桶的最大样本数量，若桶太大，程序会自动根据code_num将桶拆成多次算子下发计算距离。与<a href="#section51314823813">IVFSP训练算子模型文件生成</a>中的&lt;codebook_batch_size&gt;保持一致，默认值为“32768”。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认值为“16”。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组用于IVFSP检索时的AI Core算子模型文件，用户需要自行修改命令中的参数。例如对于Atlas 推理系列产品 生成256维，压缩后维度为32，聚类中心为1024，桶数量为32，样本数量为32768的算子：python3 ivfsp_generate_model.py -d 256 -nonzero_num 32 -nlist 1024 -handle_batch 32 -code_num 32768</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 当dim ∈ {64, 128, 256}时，k∈ {256, 512, 1024, 2048, 4096, 8192, 16384}；当dim ∈ {512, 768}时，k∈ {256, 512, 1024, 2048}。<br>● low_dim需为16的倍数且小于等于min(128, dim)。<br>● handle_batch需为16的倍数，且16 ≤ handle_batch ≤ 240。<br>● 1 ≤ pool_size ≤ 32。</td></tr>
</tbody></table>

**IVFSP AICPU算子模型文件生成<a id="section10476137113814"></a>**

<a name="table1844216303913"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 ivfsp_aicpu_generate_model.py --cores &lt;core_num&gt; -p &lt;process_id&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;core_num&gt;：AI Core的个数，默认值为“8”，无需设置。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组用于IVFSP检索时的AICPU算子模型文件。例如对于Atlas 推理系列产品 生成aicpu算子：python3 ivfsp_aicpu_generate_model.py --cores 8。</td></tr>
</tbody></table>

**IVFSP训练算子模型文件生成<a id="section51314823813"></a>**

<a name="table142311552394"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 ivfsp_generate_pyacl_model.py --cores &lt;core_num&gt; -d &lt;dim&gt; -nonzero_num &lt;low_dim&gt; -nlist &lt;k&gt; -batch_size &lt;batch_size&gt; -code_num &lt;codebook_batch_size&gt; -p &lt;process_id&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;core_num&gt;：AI Core的个数，默认值为“8”，无需设置。<br>&lt;dim&gt;：特征向量维度，默认值为“256”。<br>&lt;low_dim&gt;：特征向量压缩后非零维度个数，默认值为“32”。<br>&lt;k&gt;：簇聚类中心个数。与<a href="#section11272703813">IVFSP业务算子模型文件生成</a>中的&lt;k&gt;保持一致，默认值为“1024”。<br>&lt;batch_size&gt;：训练时以batch_size大小执行训练，默认值为“32768”。<br>&lt;codebook_batch_size&gt;：训练时每次最大按codebook_batch_size样本数量操作码本，必须为2的幂次。与<a href="#section11272703813">IVFSP业务算子模型文件生成</a>中的&lt;code_num&gt;保持一致，默认值为“32768”。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组用于IVFSP检索时的算子模型文件，用户需要自行修改命令中的参数。生成的IVFSP训练算子模型文件，保存在当前目录的子目录op_models_pyacl下。例如对于Atlas 推理系列产品 生成256维 压缩后维度为32，nlist聚类中心1024，查询数量为32768，样本数量为32768算子：python3 ivfsp_generate_pyacl_model.py -d 256 -nonzero_num 32 -nlist 1024 -batch_size 32768 -code_num 32768</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 当dim ∈ {64, 128, 256}时，k∈ {256, 512, 1024, 2048, 4096, 8192, 16384}；当dim ∈ {512, 768}时，k∈ {256, 512, 1024, 2048}。<br>● low_dim需为16的倍数且小于等于min(128, dim)。<br>● batch_size需为16的倍数。<br>● codebook_batch_size需为16的倍数。</td></tr>
</tbody></table>

#### VSTAR<a name="ZH-CN_TOPIC_0000002044867041"></a>

VSTAR检索当前只支持<term>Atlas 推理系列产品</term>，涉及VSTAR业务算子模型文件（vstar\_generate\_models.py）生成，具体请参见[VSTAR](#vstar)。

算子生成环境需要跟码本生成保持一致，具体请参见[总体说明](#总体说明)。

**VSTAR业务算子模型文件生成<a name="section11272703813"></a>**

<a name="table4331184817108"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 vstar_generate_models.py --dim &lt;dim&gt; --nlistL1 &lt;nlist1&gt; --subDimL1 &lt;sub_dim1&gt; --nProbeL1 &lt;nprobe1&gt; --nProbeL2 &lt;nprobe2&gt; --segmentNumL3 &lt;segment&gt; --pool &lt;pool_size&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;dim&gt;：特征向量维度，默认值为“256”。<br>&lt;nlist1&gt;：一级簇聚类中心个数。默认值为“1024”。<br>&lt;nprobe1&gt;：检索时每次下发计算时的一级候选桶数量，默认值为“[72]”。<br>&lt;nprobe2&gt;：检索时每次下发计算时的二级候选桶数量，默认值为“[64, 296]”。<br>&lt;sub_dim1&gt;：检索时一级降维后的维度大小，默认值为“32”。<br>&lt;segment&gt;：检索时从nprobe2中用于搜索数据段数，默认值“[512, 1000, 1504]”。<br>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认“16”。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组用于VSTAR检索时的AI Core和AICPU算子模型文件，用户需要自行修改命令中的参数。例如对于Atlas 推理系列产品 生成256维 nlist聚类中心为1024，一级候选桶nprobe1为72，二级候选桶nprobe2为64，降维后32，搜索段segment512的算子：python3 vstar_generate_models.py --dim 256 --nlistL1 1024 --subDimL1 32 --nProbeL1 72 --nProbeL2 64 --segmentNumL3 512</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dim ∈ {128, 256, 512, 1024}。<br>● nlist1 ∈ {256, 512, 1024}。<br>● sub_dim1 ∈ {32，64，128}。sub_dim1必须小于dim。<br>● nprobe1 ∈ (16, nlist1]。nprobe1是int类型的列表，且列表中的数值必须是8的整数倍。<br>● nprobe2 ∈ [16, nprobe1 * n]。当dim为1024时n为16，其余维度n为32，nprobe2是int类型的列表，且列表中的数值必须是8的整数倍。<br>● segment ∈ (100, 5000]。segment是int类型的列表，且segment必须是8的整数倍。<br>● pool_size∈[1, 32]。运行脚本前请先确定宿主机最大能支持的进程数量合理设置。</td></tr>
</tbody></table>

**涉及算法<a name="section16686174317488"></a>**

[AscendIndexVStar](./api/02_approximate_retrieval/11_AscendIndexVStar.md#ascendindexvstar)

[AscendIndexGreat](./api/02_approximate_retrieval/12_AscendIndexGreat.md#ascendindexgreat)

#### IVFFLAT<a name="ZH-CN_TOPIC_0000002478096638"></a>

<a name="table4331184817108"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 ivfflat_generate_model.py -d &lt;dim&gt; -c &lt;coarse_centroid_num&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;dim&gt;：特征向量维度，默认值为“128”。<br>&lt;coarse_centroid_num&gt;：一级簇聚类中心个数。默认值为“1024”。<br>&lt;core_num&gt;：昇腾AI处理器AI Core的个数，默认为“40”。不指定该值时，根据&lt;npu_type&gt;配置：当&lt;npu_type&gt;配置为910B3时，&lt;core_num&gt;配置为40。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认值为“10”。<br>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持Atlas A2 推理系列产品，Atlas A3 推理系列产品和Ascend 950 系列产品，默认值为“910B4”。如果无法确定具体的npu_type，则在安装昇腾AI处理器的服务器执行<strong><code>npu-smi info</code></strong>命令进行查询，将查询到的“Name”最后一位数字删除，即是npu_type的取值。对于Atlas 800I A3 超节点服务器，可以通过<strong><code>npu-smi info -t board -i 0 -c 0</code></strong>命令进行查询，获取NPU Name信息，910_NPU Name即是npu_type的取值。对于 Ascend 950 超节点服务器请将npu_type设置为“Ascend950PR”。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组算子模型文件，用户需要自行修改命令中的参数。例如对于Atlas 800I A2 生成256维，聚类中心nlist为1024算子：python3 ivfflat_generate_model.py -c 1024 -t 910B4</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dim ∈ {64, 128, 256, 384, 512}。<br>● &lt;coarse_centroid_num&gt; ∈ {1024, 2048, 4096, 8192, 16384, 32768}<br>● 1 ≤ &lt;pool_size&gt; ≤ 32</td></tr>
</tbody></table>

**涉及算法<a name="section16686174317488"></a>**

[AscendIndexIVFFlat](./api/02_approximate_retrieval/14_AscendIndexIVFFlat.md#ascendindexivfflat)

#### IVFPQ<a name="ZH-CN_TOPIC_0000002478096638"></a>

<a name="table4331184817108"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 ivfpq_generate_model.py -d &lt;dim&gt; -c &lt;nlist&gt; --cores &lt;core_num&gt; -m &lt;m&gt; -n &lt;nbit&gt; -topK &lt;topK&gt; -b &lt;blockNum&gt; -p &lt;process_id&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;dim&gt;：特征向量维度，默认值为“128”。<br>&lt;nlist&gt;：一级簇聚类中心个数。默认值为“1024”。<br>&lt;core_num&gt;：昇腾AI处理器AI Core的个数，默认为“40”。不指定该值时，根据&lt;npu_type&gt;配置。<br>&lt;m&gt;：子空间个数，默认值为“4”。<br>&lt;nbit&gt;：每个子空间量化中心比特数，默认值为“8”，无需设置。同时会决定码本聚类中心数量ksub = 1 &lt;&lt; nbit，当nbit为8时，ksub为256<br>&lt;topK&gt;：针对每条查询向量所返回的最相近候选向量的个数，默认值为“320”，无需设置。<br>&lt;blockNum&gt;：所处理候选向量block的个数，默认值为“128”，无需设置。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;仅支持Ascend950 系列产品，默认值为“Ascend950PR”，无需设置。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组算子模型文件，用户需要自行修改命令中的参数。例如生成128维，聚类中心nlist为1024，子空间个数4，比特数8算子：python3 ivfpq_generate_model.py -d 128 -c 1024 -m 4 -n 8</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dim ∈ {128}<br>● nlist ∈ {1024, 2048, 4096, 8192, 16384, 262144, 524288}<br>● m ∈ {2, 4, 8, 16, 32}<br>● n ∈ {8}</td></tr>
</tbody></table>

**涉及算法<a name="section16686174317488"></a>**

[AscendIndexIVFPQ](./api/02_approximate_retrieval/15_AscendIndexIVFPQ.md#ascendindexivfpq)

#### IVFRaBitQ<a name="ZH-CN_TOPIC_0000002513317244"></a>

<a name="table1844216303913"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 ivfrabitq_generate_model.py -d &lt;dim&gt; -c &lt;coarse_centroid_num&gt; --cores &lt;core_num&gt; -p &lt;process_id&gt; -pool &lt;pool_size&gt; -t &lt;npu_type&gt; -m &lt;metric_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;dim&gt;：特征向量维度，默认值为“128”。<br>&lt;coarse_centroid_num&gt;：一级簇聚类中心个数。默认值为“16384”。<br>&lt;core_num&gt;：昇腾AI处理器AI Core的个数，默认为“40”。不指定该值时，根据&lt;npu_type&gt;配置：当&lt;npu_type&gt;配置为910B3时，&lt;core_num&gt;配置为40。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为“0”，无需设置。<br>&lt;pool_size&gt;：批量生成算子多进程调度的进程池大小，默认值为“10”。<br>&lt;npu_type&gt;：硬件形态，当前&lt;npu_type&gt;支持Atlas A2 推理系列产品，Atlas A3 推理系列产品，默认值为“910B4”。如果无法确定具体的npu_type，则在安装昇腾AI处理器的服务器执行<strong><code>npu-smi info</code></strong>命令进行查询，将查询到的“Name”最后一位数字删除，即是npu_type的取值。对于Atlas 800I A3 超节点服务器，可以通过<strong><code>npu-smi info -t board -i 0 -c 0</code></strong>命令进行查询，获取NPU Name信息，910_NPU Name即是npu_type的取值。<br>&lt;metric_type&gt;：向量计算方式，用于显式指定使用“L2”还是“IP”距离进行计算，默认为“L2”。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组算子模型文件，用户需要自行修改命令中的参数。例如对于Atlas 800I A2 生成128维，聚类中心nlist1024，L2距离算子：python3 ivfrabitq_generate_model.py -d 128 -c 1024 -t 910B4 -m L2</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● dim ∈ {128}<br>● &lt;coarse_centroid_num&gt; ∈ {1024, 2048, 4096, 8192, 10048, 16384, 32768}<br>● 1 ≤ &lt;pool_size&gt; ≤ 32</td></tr>
</tbody></table>

**涉及算法<a name="section16686174317488"></a>**

[AscendIndexIVFRaBitQ](./api/02_approximate_retrieval/16_AscendIndexIVFRaBitQ.md#ascendindexivfrabitq)

**运行时诊断（开发调试）<a name="ivfrabitq-runtime-debug-ref"></a>**

排查 coarse centroid 上传或 L1 粗排异常时，可通过调试环境变量分阶段定位故障点（默认关闭，不影响性能）。环境变量说明见《[附录](./09_appendix.md#ivfrabitq-debug-env)》，操作步骤与日志解读见《[常用操作 — IVFRaBitQ 运行时诊断](./08_common_operations.md#ivfrabitq-runtime-debug)》。

#### Cagra<a name="ZH-CN_TOPIC_0000002513317245"></a>

<a name="table_cagra_op"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">用法</td><td valign="middle"><strong><code>python3 cagra_generate_model.py -d &lt;dim&gt; -data_base &lt;data_base&gt; -degree &lt;degree&gt; -topK &lt;topK&gt; -p &lt;process_id&gt; -t &lt;npu_type&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;dim&gt;：特征向量维度，默认值为&quot;128&quot;。<br>&lt;data_base&gt;：底库数据量，默认值为&quot;1000000&quot;。<br>&lt;degree&gt;：图度数，默认值为&quot;64&quot;。<br>&lt;topK&gt;：检索返回的最近邻个数，默认值为&quot;64&quot;。<br>&lt;process_id&gt;：批量生成算子多进程调度的进程ID，默认值为&quot;0&quot;，无需设置。<br>&lt;npu_type&gt;：硬件形态，默认值为&quot;Ascend950PR&quot;。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">说明</td><td valign="middle">执行此命令，用户可以得到一组算子模型文件，用户需要自行修改命令中的参数。例如对于Ascend 950 系列产品 生成128维，底库100万，图度数64，topK 64的算子：python3 cagra_generate_model.py -d 128 -data_base 1000000 -degree 64 -topK 64 -t Ascend950PR</td></tr>
<tr><td width="140" align="center" valign="middle">约束说明</td><td valign="middle">● 仅支持Ascend 950 系列产品<br>● dim ∈ {64, 128, 256, 512}<br>● degree ∈ {64, 128, 256, 512}</td></tr>
</tbody></table>

##### Cagra构图脚本<a name="section_cagra_build"></a>

**环境配置**

环境依赖库参见如下：

- joblib（version ≥ 1.3.0）

可通过**pip install**命令安装，命令执行参考如下。

```bash
pip install joblib
```

**训练脚本执行**

Cagra构图脚本 "graph_build.py" 用于构建CAGRA图检索算法所需的图文件（脚本位于安装目录下的"tools/train"文件夹中）。

<a name="table_cagra_build"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">命令参考</td><td valign="middle"><strong><code>python3 graph_build.py --input_filepath &lt;input_filepath&gt; --output_filepath &lt;output_filepath&gt; --graph_degree &lt;graph_degree&gt; --intermediate_degree &lt;intermediate_degree&gt; --nn_descent_niter &lt;nn_descent_niter&gt; --eval_samples &lt;eval_samples&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;input_filepath&gt;：输入目录路径，需包含&quot;sift_base.fvecs&quot;（底库数据）和&quot;sift_query.fvecs&quot;（查询数据）。该参数为必填项。<br>&lt;output_filepath&gt;：输出目录路径，生成的KNN图文件存储在该目录下。该参数为必填项。<br>&lt;graph_degree&gt;：最终图的出度，建议与搜索算子的GRAPH_DEGREE保持一致。类型为int，默认值为&quot;64&quot;。要求大于0。<br>&lt;intermediate_degree&gt;：中间图的出度，必须大于等于&lt;graph_degree&gt;。若该值不是32的倍数，将自动向上取整为最近的32的倍数。类型为int，默认值为&quot;128&quot;。<br>&lt;nn_descent_niter&gt;：NN-Descent迭代次数。迭代次数越大，图质量越高但构建时间越长。建议设置足够的迭代次数使得R@1/10/32/64均不低于0.995。类型为int，默认值为&quot;10&quot;。要求大于0。<br>&lt;eval_samples&gt;：评估图质量时使用的采样点数。设为&quot;0&quot;则跳过评估。类型为int，默认值为&quot;1000&quot;。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">使用说明</td><td valign="middle">● 执行此命令，在&lt;output_filepath&gt;对应的目录下生成以下文件：knn_graph.bin（KNN图文件）、data_ptr.bin（底库数据文件）、visited_map.bin（访问标记文件）和queries.bin（查询数据文件）。<br>● 当输出文件存在时，将执行覆盖写，此种情况程序执行用户应该是文件的属主。<br>● &lt;intermediate_degree&gt;必须大于等于&lt;graph_degree&gt;，否则程序将报错。<br>● 构建过程中会打印每次迭代后的R@1/10/32/64召回率，建议持续迭代直至召回率稳定（建议R@1/10/32/64均不低于0.995）。<br>● 构图过程中会占用较多CPU和内存资源，建议在内存充足的环境下执行。</td></tr>
<tr><td width="140" align="center" valign="middle">调用示例</td><td valign="middle">python3 graph_build.py --input_filepath /home/user/data/sift_origin --output_filepath /home/user/output/iter_64_192 --graph_degree 64 --intermediate_degree 128 --nn_descent_niter 10</td></tr>
</tbody></table>

**涉及算法<a name="section_cagra_algo"></a>**

[AscendIndexCagra](./api/02_approximate_retrieval/18_AscendIndexCagra.md#ascendindexcagra)

#### VSTAR生成码本文件<a name="ZH-CN_TOPIC_0000002008789068"></a>

##### 总体说明<a name="ZH-CN_TOPIC_0000002045184529"></a>

**环境配置<a name="section12757124191817"></a>**

环境依赖库参见如下：

- nnae（version \>= 8.0.0, 8.5.0 及以后由 toolkit 包收编）
- python（version \>= 3.9）
- torch（version \>= 2.0.1）
- torch\_npu（version \>= 2.0.1.post4）

- numpy（version \>= 1.26.4）
- scikit-learn（version \>= 1.4.1.post1）
- tqdm（version \>= 4.66.1）

torch、TorchNPU、numpy、scikit-learn和tqdm可通过**pip install**命令安装，执行命令参考如下。

```bash
pip install numpy tqdm scikit-learn torch_npu torch
```

CANN 8.5.0之前版本需要单独安装nnae。具体安装步骤如下：

1. 下载[nnae](https://www.hiascend.com/developer/download/community/result?module=cann&product=2&model=17)软件包。
2. 执行如下命令，增加可执行权限。

    ```bash
    chmod u+x ./Ascend-cann-nnae_{version}_linux-{arch}.run
    ```

3. 执行如下命令，进行安装。

    ```bash
    ./Ascend-cann-nnae_{version}_linux-{arch}.run --install
    ```

4. 按照安装提示信息设置环境变量。

    ```bash
    source /{nnae_installation_path}/nnae/set_env.sh
    ```

**注意事项<a name="section15462185871819"></a>**

- 若import torch，TorchNPU遇到下面的错误：

    ```text
    .../libgomp.so: cannot allocate memory in static TLS block
    ```

    请执行export LD\_PRELOAD=.../libgomp.so（报错中出现的libgomp.so路径）

- 若安装numpy出现pip无法安装如下依赖时：

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

    请执行以下命令。

    ```bash
    pip install attrs cloudpickle decorator jinja2 ml-dtypes psutil scipy tornado absl-py
    ```

- 若训练码本遇到以下问题：

    ```text
    OpenBLAS warning: precompiled NUM_THREADS exceeded, adding auxiliary array for thread metadata.
    Segmentation fault (core dumped)
    ```

    请执行：

    ```bash
    export OPENBLAS_NUM_THREADS=1
    ```

    该环境变量可能影响性能，码本训练完成后，建议设置回预设值。

- --useOfflineCompile选项详细说明：

    在线算子编译耗时相比离线算子编译耗时较长。--useOfflineCompile选项用于控制是否使用离线算子编译，使用预先编译好的离线算子包执行。该方式需要用户提前安装单算子包。算子包安装指导如下：

    1. 下载[算子软件包](https://www.hiascend.com/developer/download/community/result?module=cann&product=2&model=17)。
    2. 执行如下命令，增加可执行权限。
        - CANN  8.5.0之前版本

            ```bash
            chmod u+x ./Ascend-cann-kernels-{chip_type}_{version}_linux-{arch}.run
            ```

        - CANN 8.5.0及之后版本

            ```bash
            chmod u+x ./Ascend-cann-{chip_type}-ops_{version}_linux-{arch}.run
            ```

    3. 执行如下命令，进行安装。
        - CANN 8.5.0之前版本

            ```bash
            ./Ascend-cann-kernels-{chip_type}_{version}_linux-{arch}.run --install
            ```

        - CANN 8.5.0及之后版本

            ```bash
            ./Ascend-cann-{chip_type}-ops_{version}_linux-{arch}.run --install
            ```

    4. 按照安装提示信息设置环境变量。
        - CANN 8.5.0之前版本

            ```bash
            source /{kernels_installation_path}/kernels/set_env.sh
            ```

        - CANN 8.5.0及之后版本

            ```bash
            source /usr/local/Ascend/cann/set_env.sh
            ```

##### 码本训练脚本<a name="ZH-CN_TOPIC_0000002008865568"></a>

训练涉及“vstar\_train\_codebook.py”脚本（训练脚本位于安装目录下的“tools/train”文件夹中），注意Python版本为3.9。

<a name="table48723587152"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">命令参考</td><td valign="middle"><strong><code>python3 vstar_train_codebook.py --dataPath &lt;data_path&gt; --dim &lt;dim&gt; --codebookPath &lt;codebook_output_dir&gt; --nlistL1 &lt;nlist1&gt; --subDimL1 &lt;sub_dim1&gt; --device &lt;device&gt; --batchSize &lt;batch_size&gt; --sample &lt;sample&gt; --useOfflineCompile</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;data_path&gt;：需要训练码本的原始数据路径，需要保证数据真实存在。该参数为必填项。<br>&lt;dim&gt;：特征向量维度。与VSTAR训练算子模型文件生成的&lt;dim&gt;保持一致，默认值为“256”。<br>&lt;codebook_output_dir&gt;：最终生成的码本文件所存储的路径，生成的码本文件输出到的目录，用户应该保证此目录存在，且程序的执行用户对此目录具有写权限。出于安全加固考虑，目录层级中不能含有软链接。<br>&lt;nlist1&gt;：一级簇聚类中心个数。与VSTAR训练算子模型文件生成的&lt;nlist1&gt;保持一致，默认值为“1024”。<br>&lt;sub_dim1&gt;：检索时一级降维后的维度大小，与VSTAR训练算子模型文件生成的&lt;sub_dim1&gt;保持一致，默认值为“32”。<br>&lt;device&gt;：设备逻辑ID，在指定的Device上执行训练，默认值为“1”。<br>&lt;batch_size&gt;：训练时以batch_size大小执行训练，参数范围(0，10240]，默认值为“10240”。<br>&lt;sample&gt;：训练用原始样本的采样率，0 &lt; sample ≤ 1.0，默认为“1.0”。<br>--useOfflineCompile：控制是否选择依赖算子包，使用离线算子编译，以获得性能提升。默认不开启。若开启，在命令行结尾增加该选项即可。详细说明请参见:VSTAR生成码本文件-总体说明- --useOfflineCompile选项详细说明。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">使用说明</td><td valign="middle">● &lt;data_path&gt;原始数据大小需≤一千万1024维数据，即10,000,000 * 1024 * 4 = 40,960,000,000。<br>● 执行此命令，在&lt;codebook_output_dir&gt;对应的目录下生成新目录codebook_&lt;dim&gt;_&lt;nlist1&gt;_&lt;sub_dim1&gt;.bin，即为AscendIndexVStar和AscendIndexGreat所需使用到的码本文件。<br>● 当码本文件存在时，将执行覆盖写，此种情况程序执行用户应该是文件的属主。<br>● 在执行训练生成码本前，请先参考VSTAR，生成训练算子模型文件。</td></tr>
</tbody></table>

#### （可选）Python方式生成码本文件<a name="ZH-CN_TOPIC_0000001649848464"></a>

##### IVFSP训练脚本<a name="ZH-CN_TOPIC_0000001585736180"></a>

**环境配置**

环境依赖库参见如下：

- numpy（version \> 1.16.0）
- tqdm（version ≥ 4.65.0）
- faiss-cpu（version = 1.10.0）

可通过**pip install**命令安装，命令执行参考如下。

```bash
pip install numpy tqdm faiss-cpu==1.10.0
```

执行训练脚本前，先执行如下命令设置环境变量。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**训练脚本执行**

Index SDK提供两种训练脚本方式：

- 使用IVFSP算法的[trainCodeBook接口](./api/02_approximate_retrieval/05_AscendIndexIVFSP.md#traincodebook接口)进行训练（推荐使用该方式）。
- 使用“ivfsp\_train\_codebook.py”脚本进行训练。训练脚本位于安装目录下的“tools/train”文件夹中，注意Python版本为3.9.11。为了用户执行方便，提供了“ivfsp\_train\_codebook\_example.sh”样例脚本（脚本位于安装目录下的“tools/train”文件夹中），用户可在此文件上根据实际场景修改参数值，然后执行此脚本生成码本文件。

<a name="table48723587152"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">命令参考</td><td valign="middle"><strong><code>python3 ivfsp_train_codebook.py --dim &lt;dim&gt; --nonzero_num &lt;nonzero_num&gt; --nlist &lt;nlist&gt; --num_iter &lt;num_iter&gt; --device &lt;device&gt; --batch_size &lt;batch_size&gt; --code_num &lt;code_num&gt; --ratio &lt;ratio&gt; --learn_data_path &lt;learn_data_path&gt; --codebook_output_dir &lt;codebook_output_dir&gt; --train_model_dir &lt;train_model_dir&gt;</code></strong></td></tr>
<tr><td width="140" align="center" valign="middle">参数名称</td><td valign="middle">&lt;dim&gt;：特征向量维度。与IVFSP训练算子模型文件生成的&lt;dim&gt;保持一致，要求大于0。<br>&lt;nonzero_num&gt;：特征向量压缩后非零维度个数，与IVFSP训练算子模型文件生成的&lt;low_dim&gt;保持一致，要求大于0。<br>&lt;nlist&gt;：簇聚类中心个数。与IVFSP训练算子模型文件生成的&lt;k&gt;保持一致，要求大于0。<br>&lt;num_iter&gt;：训练迭代次数参数，默认为20。迭代次数设置过大，会导致训练时长增加，要求大于0。<br>&lt;device&gt;：设备逻辑ID，在指定的Device上执行训练，默认值为“0”。<br>&lt;batch_size&gt;：训练时以batch_size大小执行训练。与IVFSP训练算子模型文件生成的&lt;batch_size&gt;保持一致，要求大于0，小于等于32768，默认值为“32768”。<br>&lt;code_num&gt;：每次最大按code_num样本数量操作码本，必须为2的幂次。与IVFSP训练算子模型文件生成的&lt;codebook_batch_size&gt;保持一致，要求大于0，小于等于32768，默认值为“32768”。<br>&lt;ratio&gt;：训练用原始样本的采样率，0 &lt; ratio ≤ 1.0，默认为1.0。<br>&lt;learn_data_path&gt;：训练用的原始特征文件路径，支持bin、npy格式，bin存储方式为行优先，数据类型为float32。<br>&lt;codebook_output_dir&gt;：生成的码本文件输出到的目录，用户应该保证此目录存在，且程序的执行用户对此目录具有写权限；出于安全加固的考虑，此目录层级中不能含有软链接。<br>&lt;train_model_dir&gt;：IVFSP训练算子模型文件所在目录。<br>--help | -h：查询帮助信息。</td></tr>
<tr><td width="140" align="center" valign="middle">使用说明</td><td valign="middle">● 执行此命令，在&lt;codebook_output_dir&gt;对应的目录下生成文件codebook_&lt;dim&gt;_&lt;nonzero_num&gt;_&lt;nlist&gt;.bin和codebook_&lt;dim&gt;_&lt;nonzero_num&gt;_&lt;nlist&gt;.npy，codebook_&lt;dim&gt;_&lt;nonzero_num&gt;_&lt;nlist&gt;.bin即为AscendIndexIVFSP所需使用到的码本文件。<br>● 当码本文件存在时，将执行覆盖写，此种情况程序执行用户应该是文件的属主。<br>● 在执行训练生成码本前，请先参考IVFSP训练算子模型文件生成，生成训练算子模型文件。<br>● learn_data_path指定的数据大小必须大于等于nonzero_num * nlist * sizeof(float32) 字节。</td></tr>
</tbody></table>

##### 降维训练脚本<a name="ZH-CN_TOPIC_0000001681635905"></a>

**环境依赖<a name="section162431329141010"></a>**

- 安装Python3.9（支持Python3.9、Python3.10和Python3.11，推荐使用Python3.9）。
- 安装Faiss 1.10.0。可通过**pip install**命令安装，命令执行参考如下。

    ```bash
    pip install faiss-cpu==1.10.0
    ```

- 安装torch\_cpu和TorchNPU。安装方法参见[链接](https://gitcode.com/Ascend/pytorch)。请根据版本配套表，选择对应版本安装。

**训练模型<a name="section8422152014206"></a>**

本章节涉及的脚本的默认存放路径为：“tools/train/reduction”。

1. 训练模型。

    ```bash
    python3 call_train.py --dataset_dir=Dataset_Dir --val_dataset_dir=./valid --generate_val=True --save_path=./modelsDr --dim=512 --npu=0 --ratio=4 --metric=L2 --mode=train --train_size=100000 --epochs=20 --train_batch_size=8192 --infer_batch_size=128 --learning_rate=0.0005 --log_stride=500 --construct_neighbors=100 --queries_validation=1000
    ```

    |参数|说明|
    |--|--|
    |dataset_dir|数据集路径，类型为string，必须设置。目前实现默认读取base.npy，query.npy和gt.npy。若数据集为其他名称，可以自行实现数据集读取，并对该脚本get_train_data所在行做对应修改。例如。原代码为：<br>```# load dataset demo before training, modify here if you want to load your own dataset        #####################################################################        learn, base = get_train_data(args.dataset_dir, args.train_size)        #####################################################################```    <br>可修改为：<br>```# load dataset demo before training, modify here if you want to load your own dataset        #####################################################################        # learn, base = get_train_data(args.dataset_dir, args.train_size)        learn = np.fromfile(YOUR_LEARN_DATASET_DIR, dtype=np.float32).reshape((-1, YOUR_DATA_DIM))        base = np.fromfile(YOUR_BASE_DATASET_DIR, dtype=np.float32).reshape((-1, YOUR_DATA_DIM))        #####################################################################```|
    |val_dataset_dir|generate_val为True时有效，生成验证集的存放路径，类型为string，默认值为./validation/。|
    |generate_val|是否生成验证集。首次训练请设置为True。类型为bool，默认为False。|
    |save_path|模型存放路径。类型为string，必须设置。|
    |dim|可选，数据集维度。取值范围：[96, 128, 200, 256, 512, 2048]。类型为int，默认值为512。|
    |npu|训练所用的DeviceId，即设备号。类型为int。仅支持单卡训练，不指定时默认使用CPU训练。|
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

2. 生成OM模型。

    执行训练脚本前，先执行如下命令设置环境变量（根据CANN软件包的实际安装路径修改）。

    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
    ```

    1. 生成精度为32的om模型。

        ```bash
        bash atc.sh {save_path} {om_name} {input_shape}
        ```

    2. 生成精度为16的om模型

        ```bash
        bash atc_16.sh {save_path} {om_name} {input_shape}
        ```

    - \{save\_path\}：必选，表示模型存储的路径。路径中文件名需要以".onnx"或".pb"结尾，否则脚本会获取环境变量"framework"、"input\_format"等值，导致脚本执行异常。
    - \{om\_name\}：可选，表示生成OM模型的名字，默认与onnx模型名字相同。
    - \{input\_shape\}：可选，默认为onnx模型的输入维度，格式为actual\_input\_1:infer\_batch\_size,dim，建议使用默认值，不建议修改。
    - **bash atc.sh**和**bash atc\_16.sh**仅支持<term>Atlas 推理系列产品</term>。
