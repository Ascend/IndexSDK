# examples

## 介绍

**本仓库提供了昇腾Index SDK组件实现的几种常见检索算法的demo**

## gtest安装教程

部分用例需要安装gtest：

```bash
wget https://github.com/google/googletest/archive/refs/tags/release-1.8.1.tar.gz && \
tar xf release-1.8.1.tar.gz && cd googletest-release-1.8.1 && \
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local/gtest . && make -j && make install && \
cd .. && rm -rf release-1.8.1.tar.gz googletest-release-1.8.1
```

## 代码目录结构说明

```bash
.
|-- build.sh
|-- CMakeLists.txt
|-- README.md
|-- TestAscendIReduction.cpp                       # 降维算法NN降维Pcar降维demo
|-- TestAscendIndexAggressTs.cpp                   # 时空库IP距离，带属性过滤 支持组batch demo
|-- TestAscendIndexBinaryFlat.cpp                  # 二值化底库特征汉明距离BinaryFlat算法demo
|-- TestAscendIndexCluster.cpp                     # FP32 聚类场景AscendCluster算法demo
|-- TestAscendIndexFlat.cpp                        # FP32转FP16 Flat算法暴搜demo
|-- TestAscendIndexGreat.cpp                       # Great近似检索算法demo
|-- TestAscendIndexIVFSP.cpp                       # IVFSP近似检索算法demo
|-- TestAscendIndexIVFSQ.cpp                       # IVFSQ 近似检索算法demo
|-- TestAscendIndexIVFSQT.cpp                      # IVFSQT近似检索算法demo
|-- TestAscendIndexIVFSQTwithCpuFlat.cpp           # IVFSQT粗搜加cpu精搜demo
|-- TestAscendIndexInt8Flat.cpp                    # 底库数据为int8 int8Flat算法暴搜demo
|-- TestAscendIndexInt8FlatWithCPU.cpp             # 底库数据为int8 int8Flat算法CPU同步落盘 demo
|-- TestAscendIndexInt8FlatWithReduction.cpp       # FP32 降维量化为Int8后，int8Flat算法暴搜demo
|-- TestAscendIndexInt8FlatWithSQ.cpp              # FP32 SQ 量化为int8后， 暴搜demo
|-- TestAscendIndexSQ.cpp                          # FP32 SQ 量化为Int8后，反量化SQ算法暴搜demo
|-- TestAscendIndexSQMulPerformance.cpp            # 布控库 IP距离 SQ算法demo
|-- TestAscendIndexTS.cpp                          # 时空库，汉明距离，带属性过滤demo
|-- TestAscendIndexTS_int8Cos.cpp                  # 时空库，int8 cos距离，带属性过滤demo
|-- TestAscendIndexVStar.cpp                       # VStar近似检索算法demo
`-- TestAscendMultiSearch.cpp                      # 多Index批量检索demo
```

注：<br>
```TestAscendIndexIVFSP.cpp``` 需要根据实际情况填写数据集（特征数据、查询数据、groundtruth数据）、码本，所在的目录。<br>
```TestAscendIndexGreat.cpp``` 需要根据实际情况填写数据集（特征数据、查询数据、groundtruth数据）、码本，所在的目录。<br>
```TestAscendIndexVStar.cpp``` 需要根据实际情况填写数据集（特征数据、查询数据、groundtruth数据）、码本，所在的目录。<br>
```TestAscendIReduction.cpp``` 需要根据实际情况填写对应的NN降维模型所在的目录。<br>

## Demo使用说明

1. **请先正确安装Index SDK组件及其依赖的driver、firmware、Ascend toolkit、OpenBLAS、Faiss**

2. 执行build.sh编译demo

    ```bash
    # 查看帮助和可用测试用例
    bash build.sh -h

    # 编译所有测试用例
    bash build.sh

    # 编译指定测试用例（按名称）
    bash build.sh TestAscendIndexFlat

    # 编译指定测试用例（按编号）
    bash build.sh 1
    ```

    也可以使用命令行编译：

    ```bash
    g++ -std=c++11 -march=armv8-a -fPIC -fstack-protector-all \
        -Wno-sign-compare -D_FORTIFY_SOURCE=2 -O3 -Wall -Wextra \
        -DFINTEGER=int -fopenmp \
        -o TestAscendIndexFlat TestAscendIndexFlat.cpp \
        -I/usr/local/Ascend/mxIndex/include \
        -I/usr/local/faiss/faiss1.10.0/include \
        -I/usr/local/gtest/include \
        -I/usr/local/Ascend/driver/include/dvpp/ \
        -L/usr/local/Ascend/mxIndex/host/lib \
        -L/usr/local/faiss/faiss1.10.0/lib \
        -L/usr/local/gtest/lib \
        -L/usr/local/Ascend/driver/lib64/driver \
        -lopenblas -lfaiss -lascendfaiss -lascend_hal -lgtest
    ```

    > [!NOTE]
    > 编译 TestAscendIndexIVFRabitQ.cpp 需要安装 faiss1.14.1。

3. 设置环境变量与生成算子

    执行如下命令设置环境变量（根据CANN软件包的实际安装路径修改）：

    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export LD_LIBRARY_PATH=${MXINDEX_INSTALL_PATH}/host/lib:$LD_LIBRARY_PATH
    ```

    MXINDEX_INSTALL_PATH为Index SDK实际安装路径，本例中为```/usr/local/Ascend/mxIndex```

    生成算子：

    所有算子生成的python文件均在```${MXINDEX_INSTALL_PATH}/tools/```目录下，可执行```-h```参数 查看具体参数意义

    以TestAscendIndexFlat.cpp中需要生成的Flat算子为例, 执行：

    ```bash
    cd ${MXINDEX_INSTALL_PATH}/ops/
    bash custom_opp_{arch}.run

    cd ${MXINDEX_INSTALL_PATH}/tools/
    # 生成aicpu和flat 512维的算子
    ```

    设置算子的环境变量，并将算子移动至算子目录：

    ```bash
    export MX_INDEX_MODELPATH=/usr/local/Ascend/mxIndex-{version}/modelpath/
    mv ${INDEX_INSTALL_PATH}/tools/op_models/* ${MX_INDEX_MODELPATH}
    ```

    注意：算子环境变量请勿使用软链接，而是算子实际所在目录。

4. 在build目录中找到对应的二进制可执行文件

    以TestAscendIndexFlat.cpp为例，执行:

    ```bash
    cd examples/build/
    ./TestAscendIndexFlat
    ```
