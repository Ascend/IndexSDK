# examples

English | [简体中文](README.md)

## Introduction

**This repository provides demos of several common retrieval algorithms implemented by the Ascend Index SDK component**

## gtest Installation Guide

Some test cases require installing gtest:

```bash
wget https://github.com/google/googletest/archive/refs/tags/release-1.8.1.tar.gz && \
tar xf release-1.8.1.tar.gz && cd googletest-release-1.8.1 && \
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local/gtest . && make -j && make install && \
cd .. && rm -rf release-1.8.1.tar.gz googletest-release-1.8.1
```

## Code Directory Structure

```bash
.
|-- build.sh
|-- CMakeLists.txt
|-- README.md
|-- TestAscendIReduction.cpp                       # Dimensionality reduction algorithm NN reduction Pcar reduction demo
|-- TestAscendIndexAggressTs.cpp                   # Spatiotemporal library IP distance with attribute filtering, supporting group batch demo
|-- TestAscendIndexBinaryFlat.cpp                  # Binary base feature Hamming distance BinaryFlat algorithm demo
|-- TestAscendIndexCagra.cpp                       # CAGRA graph retrieval algorithm demo
|-- TestAscendIndexCluster.cpp                     # FP32 clustering scenario AscendCluster algorithm demo
|-- TestAscendIndexFlat.cpp                        # FP32 to FP16 Flat algorithm brute-force search demo
|-- TestAscendIndexGreat.cpp                       # Great approximate retrieval algorithm demo
|-- TestAscendIndexIVFSP.cpp                       # IVFSP approximate retrieval algorithm demo
|-- TestAscendIndexIVFSQ.cpp                       # IVFSQ approximate retrieval algorithm demo
|-- TestAscendIndexIVFSQT.cpp                      # IVFSQT approximate retrieval algorithm demo
|-- TestAscendIndexIVFSQTwithCpuFlat.cpp           # IVFSQT coarse search plus CPU fine search demo
|-- TestAscendIndexInt8Flat.cpp                    # Brute-force search demo of the int8Flat algorithm with int8 base data
|-- TestAscendIndexInt8FlatWithCPU.cpp             # Demo of the int8Flat algorithm with int8 base data and CPU synchronous write to drive
|-- TestAscendIndexInt8FlatWithReduction.cpp       # Brute-force search demo of the int8Flat algorithm after FP32 dimension reduction and quantization to int8
|-- TestAscendIndexInt8FlatWithSQ.cpp              # Brute-force search demo after FP32 SQ quantization to int8
|-- TestAscendIndexSQ.cpp                          # Brute-force search demo of the dequantization SQ algorithm after FP32 SQ quantization to int8
|-- TestAscendIndexSQMulPerformance.cpp            # Surveillance library IP distance SQ algorithm demo
|-- TestAscendIndexTS.cpp                          # Spatiotemporal library, Hamming distance, with attribute filtering demo
|-- TestAscendIndexTS_int8Cos.cpp                  # Spatiotemporal library, int8 cos distance, with attribute filtering demo
|-- TestAscendIndexVStar.cpp                       # VStar approximate retrieval algorithm demo
`-- TestAscendMultiSearch.cpp                      # Multi-Index batch retrieval demo
```

Note:<br>
In `TestAscendIndexIVFSP.cpp`, specify the directories of the dataset (feature data, query data, and ground-truth data) and codebook based on the actual situation.<br>
In `TestAscendIndexGreat.cpp`, specify the directories of the dataset (feature data, query data, and ground-truth data) and codebook based on the actual situation.<br>
In `TestAscendIndexVStar.cpp`, specify the directories of the dataset (feature data, query data, and ground-truth data) and codebook based on the actual situation.<br>
In `TestAscendIReduction.cpp`, specify the directory of the corresponding NN-based dimensionality reduction model based on the actual situation.<br>

## Demo Usage

1. **Install the Index SDK component and its dependent driver, firmware, Ascend toolkit, OpenBLAS, and Faiss correctly first.**

2. Run `build.sh` to compile the demo.

    ```bash
    # View help and available test cases
    bash build.sh -h

    # Compile all test cases
    bash build.sh

    # Compile a specified test case (by name)
    bash build.sh TestAscendIndexFlat

    # Compile a specified test case (by number)
    bash build.sh 1
    ```

    You can also compile using the command line:

    ```bash
    g++ -std=c++11 -march=armv8-a -fPIC -fstack-protector-all \
        -Wno-sign-compare -D_FORTIFY_SOURCE=2 -O3 -Wall -Wextra \
        -DFINTEGER=int -fopenmp \
        -o TestAscendIndexFlat TestAscendIndexFlat.cpp \
        -I/usr/local/Ascend/mxIndex/include \
        -I/usr/local/faiss/include \
        -I/usr/local/gtest/include \
        -I/usr/local/Ascend/driver/include/dvpp/ \
        -L/usr/local/Ascend/mxIndex/host/lib \
        -L/usr/local/faiss/lib \
        -L/usr/local/gtest/lib \
        -L/usr/local/Ascend/driver/lib64/driver \
        -lopenblas -lfaiss -lascendfaiss -lascend_hal -lgtest
    ```

    > [!NOTE]
    > Compiling `TestAscendIndexIVFRabitQ.cpp` / `TestAscendIndexIVFRaBitQSmallAdd.cpp` / `TestAscendIndexIVFRaBitQBoundary.cpp` requires faiss1.14.1.
    >
    > Small-batch insertion example: `./TestAscendIndexIVFRaBitQSmallAdd [ntotal] [nlist] [batch] [nprobe]`, with defaults `100000 1024 64 32`. To troubleshoot accelerator memory, set `ASCENDFAISS_MEM_DEBUG=1`.

3. Set environment variables and generate operators.

    Run the following command to set the environment variables (modify the path according to the actual installation path of the CANN software package):

    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export LD_LIBRARY_PATH=${MXINDEX_INSTALL_PATH}/host/lib:$LD_LIBRARY_PATH
    ```

    `MXINDEX_INSTALL_PATH` is the actual installation path of Index SDK, which is `/usr/local/Ascend/mxIndex` in this example.

    Generate the operators:

    All Python files for operator generation are located in the `${MXINDEX_INSTALL_PATH}/tools/` directory. You can run the `-h` parameter to view the meaning of specific parameters.

    Taking the Flat operator that needs to be generated in `TestAscendIndexFlat.cpp` as an example, run:

    ```bash
    cd ${MXINDEX_INSTALL_PATH}/ops/
    bash custom_opp_{arch}.run

    cd ${MXINDEX_INSTALL_PATH}/tools/
    # Generate the aicpu and flat 512-dimensional operators
    ```

    Set the environment variables for the operators and move the operators to the operator directory:

    ```bash
    export MX_INDEX_MODELPATH=/usr/local/Ascend/mxIndex-{version}/modelpath/
    mv ${INDEX_INSTALL_PATH}/tools/op_models/* ${MX_INDEX_MODELPATH}
    ```

    Note: Do not use a symbolic link for the operator environment variable. Use the actual directory where the operators are located.

4. Find the corresponding binary executable file in the build directory.

    Taking `TestAscendIndexFlat.cpp` as an example, run:

    ```bash
    cd examples/build/
    ./TestAscendIndexFlat
    ```
