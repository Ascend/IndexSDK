# This file is part of the IndexSDK project.
# Copyright (c) 2025 Huawei Technologies Co.,Ltd.
#
# IndexSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

CMAKE_MINIMUM_REQUIRED(VERSION 3.20)

PROJECT(UT)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/config)
include(env)
include(CreateTestableTarget)

ENABLE_TESTING()

FIND_PACKAGE(OpenMP REQUIRED)
IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS          "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSE()
    MESSAGE(WARNING "OPENMP NOT FOUND")
ENDIF()

OPTION(UT_COVERAGE "UT COVERAGE SWITCH" OFF)
OPTION(ASAN_OPTION "ASAN SWITCH" OFF)

IF(${UT_COVERAGE} STREQUAL "on")
    SET(GCC_COVERAGE_COMPILE_FLAGS "-g -O3 -fprofile-arcs -ftest-coverage")
    SET(GCC_COVERAGE_LINK_FLAGS    "-lgcov")

    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")
ENDIF()

MESSAGE("${BLUE}-- gtest home director : ${GTEST_HOME} ${END}")
MESSAGE("${BLUE}-- faiss home director : ${FAISS_HOME} ${END}")
MESSAGE("${BLUE}-- protobuf home director : ${PROTOBUF_HOME} ${END}")

ADD_COMPILE_OPTIONS(
    -Wall 
    -fno-strict-aliasing 
    -Wreturn-type
    -D_FORTIFY_SOURCE=2 
    -frename-registers 
    -fpeel-loops 
    -fopenmp 
    -O0
    -g
    -fno-access-control # 加入这个编译选项 可以完美访问私有成员 单独测试私有函数
    -Wno-pmf-conversions # 加入这个编译选项 可以将 类成员函数转为普通函数进行测试
    -fPIC
    -fstack-protector-all
)

IF(${ASAN_OPTION} STREQUAL "on")
    ADD_COMPILE_OPTIONS(
        -fsanitize=address
        -fsanitize=leak
        -fsanitize-recover=address,all
        -fno-omit-frame-pointer
    )
ENDIF()

ADD_DEFINITIONS(
    -DUSE_ACL_INTERFACE_V2
    -DHOSTCPU
)

SET(ASCENDFAISS_SRC
    ${ASCEND_PROJECT_HOME}/ascend/AscendCloner.cpp
    ${ASCEND_PROJECT_HOME}/ascend/AscendClonerOptions.cpp
    ${ASCEND_PROJECT_HOME}/ascend/AscendIndex.cpp
    ${ASCEND_PROJECT_HOME}/ascend/AscendIndexFlat.cpp
    ${ASCEND_PROJECT_HOME}/ascend/AscendIndexInt8.cpp
    ${ASCEND_PROJECT_HOME}/ascend/AscendIndexInt8Flat.cpp
    ${ASCEND_PROJECT_HOME}/ascend/AscendIndexIVF.cpp
    ${ASCEND_PROJECT_HOME}/ascend/AscendIndexIVFSQ.cpp
    ${ASCEND_PROJECT_HOME}/ascend/AscendIndexSQ.cpp
    ${ASCEND_PROJECT_HOME}/ascend/AscendMultiIndexSearch.cpp
    ${ASCEND_PROJECT_HOME}/ascend/AscendNNInference.cpp
    ${ASCEND_PROJECT_HOME}/ascend/custom/AscendClustering.cpp
    ${ASCEND_PROJECT_HOME}/ascend/custom/AscendIndexFlatAT.cpp
    ${ASCEND_PROJECT_HOME}/ascend/custom/AscendIndexFlatATInt8.cpp
    ${ASCEND_PROJECT_HOME}/ascend/custom/AscendIndexIVFSQT.cpp
    ${ASCEND_PROJECT_HOME}/ascend/custom/IReduction.cpp
    ${ASCEND_PROJECT_HOME}/ascend/custom/PcarReduction.cpp
    ${ASCEND_PROJECT_HOME}/ascend/custom/NNReduction.cpp
    ${ASCEND_PROJECT_HOME}/ascend/impl/AscendIndexFlatImpl.cpp
    ${ASCEND_PROJECT_HOME}/ascend/impl/AscendIndexImpl.cpp
    ${ASCEND_PROJECT_HOME}/ascend/impl/AscendIndexInt8Impl.cpp
    ${ASCEND_PROJECT_HOME}/ascend/impl/AscendIndexInt8FlatImpl.cpp
    ${ASCEND_PROJECT_HOME}/ascend/impl/AscendIndexIVFImpl.cpp
    ${ASCEND_PROJECT_HOME}/ascend/impl/AscendIndexIVFSQImpl.cpp
    ${ASCEND_PROJECT_HOME}/ascend/impl/AscendIndexSQImpl.cpp
    ${ASCEND_PROJECT_HOME}/ascend/impl/AscendNNInferenceImpl.cpp
    ${ASCEND_PROJECT_HOME}/ascend/custom/impl/AscendIndexFlatATImpl.cpp
    ${ASCEND_PROJECT_HOME}/ascend/custom/impl/AscendIndexFlatATInt8Impl.cpp
    ${ASCEND_PROJECT_HOME}/ascend/custom/impl/AscendIndexIVFSQFuzzyImpl.cpp
    ${ASCEND_PROJECT_HOME}/ascend/custom/impl/AscendIndexIVFSQCImpl.cpp
    ${ASCEND_PROJECT_HOME}/ascend/custom/impl/AscendIndexIVFSQTImpl.cpp
    ${ASCEND_PROJECT_HOME}/ascend/utils/fp16.cpp
    ${ASCEND_PROJECT_HOME}/ascend/utils/MergeResUtils.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/StandardAscendResources.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/ThreadSafeStandardAscendResources.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/impl/AuxIndexStructures.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/impl/Index.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/impl/IndexFlat.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/impl/IndexInt8.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/impl/IndexInt8Flat.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/impl/IndexIVF.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/impl/IndexIVFSQ.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/impl/Int8L2Norm.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/impl/IndexSQ.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/utils/AscendMemory.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/utils/AscendOpDesc.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/utils/AscendOperator.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/utils/AscendStackMemory.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/utils/AscendUtils.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/utils/MemorySpace.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/utils/Random.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/utils/TopkOp.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/utils/ModelInference.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/utils/ModelExecuter.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/utils/DeviceMemMng.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/utils/ThreadSafeDistComputeOpsManager.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/utils/DistComputeOpsManager.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/impl/AscendClusteringImpl.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/impl/AscendIndexBinaryFlatImpl.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/impl/TSBase.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/impl/TSBinaryFlat.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/impl/TSFlatIP.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/impl/TSInt8FlatCos.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/impl/TSInt8FlatL2.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/impl/HeteroBlockGroupMgrImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/impl/AscendIndexClusterImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/impl/AscendIndexILFlatImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index/AscendIndexCluster.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index/AscendIndexILFlat.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index/IndexFlatIPAicpu.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index/IndexFlatL2Aicpu.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index/IndexInt8FlatL2Aicpu.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index/IndexInt8FlatCosAicpu.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index/IndexIVFSQIPAicpu.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index/IndexIVFSQL2Aicpu.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index/IndexSQIPAicpu.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index/IndexSQL2Aicpu.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index/AscendIndexTS.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index/AscendIndexBinaryFlat.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index_custom/IndexFlatAT.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index_custom/IndexFlatATAicpu.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index_custom/IndexFlatATInt8Aicpu.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index_custom/IndexFlatATSubAicpu.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index_custom/IndexIVFSQCIPAicpu.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index_custom/IndexIVFSQTIPAicpu.cpp
    ${ASCEND_PROJECT_HOME}/ascendhost/src/index_custom/IndexInt8FlatApproxL2Aicpu.cpp
    ${ASCEND_PROJECT_HOME}/common/threadpool/AscendThreadPool.cpp
    ${ASCEND_PROJECT_HOME}/common/hmm/TSInt8FlatHPPCosFactoryStub.cpp
    ${CMAKE_CURRENT_LIST_DIR}/stub/hmm/HmmMock.cpp
    # for ilflat
    ${ASCEND_PROJECT_HOME}/ascenddaemon/impl_device/IndexILFlat.cpp
    ${ASCEND_PROJECT_HOME}/ascenddaemon/utils/AscendRWLock.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Common.cpp
    # for great
    ${ASCEND_PROJECT_HOME}/ascend/mixsearch/great/AscendIndexGreat.cpp
    ${CMAKE_CURRENT_LIST_DIR}/stub/great/GreatMock.cpp
    # vstar
    ${ASCEND_PROJECT_HOME}/ascend/mixsearch/vstar/AscendIndexVStar.cpp
    ${CMAKE_CURRENT_LIST_DIR}/stub/vstar/VstarMock.cpp
)

SET(ASCENDFAISS_INCLUDE
    ${PROTOBUF_HOME}/include
    ${FAISS_HOME}/include
    ${GTEST_HOME}/include
    ${CMAKE_CURRENT_LIST_DIR}/moduleUT
    ${OPEN_SOURCE_HOME}/AscendCLMock
    ${OPEN_SOURCE_HOME}/AscendCLMock/acl
    ${OPEN_SOURCE_HOME}/AscendCLMock/securec
    ${OPEN_SOURCE_HOME}/mockcpp/output/include
    ${ASCEND_PROJECT_HOME}/../
    ${ASCEND_PROJECT_HOME}/
    ${ASCEND_PROJECT_HOME}/ascend
    ${ASCEND_PROJECT_HOME}/ascend/utils
    ${ASCEND_PROJECT_HOME}/ascend/mixsearch/include
    ${ASCEND_PROJECT_HOME}/ascend/mixsearch/vstar
    ${ASCEND_PROJECT_HOME}/ascenddaemon
    ${ASCEND_PROJECT_HOME}/ascenddaemon/impl
    ${ASCEND_PROJECT_HOME}/ascenddaemon/utils
    ${ASCEND_PROJECT_HOME}/common
    ${ASCEND_PROJECT_HOME}/ascendhost/include
    ${ASCEND_PROJECT_HOME}/ascendhost/include/impl
    ${ASCEND_PROJECT_HOME}/ascendhost/include/index
    ${SECUREC_HOME}/include
    ${ASCEND_PROJECT_HOME}/ascend/mixsearch/include
    ${ASCEND_PROJECT_HOME}/ascend/mixsearch/great
    ${ASCEND_PROJECT_HOME}/ascend/mixsearch
    ${PROJECT_SOURCE_DIR}
)

SET(ASCENDFAISS_LIBRARY_PATH
    ${CMAKE_INSTALL_PREFIX}/lib
    ${GTEST_HOME}/lib
    ${FAISS_HOME}/lib
    ${PROTOBUF_HOME}/lib
    ${PROTOBUF_HOME}/lib
    ${OPEN_SOURCE_HOME}/AscendCLMock/acl/lib
    ${OPEN_SOURCE_HOME}/mockcpp/output/lib
    ${FAISS_HOME}/lib
    ${SECUREC_HOME}/lib
)

SET(ASCENDFAISS_LIBRARY
    openblas 
    faiss
    ascendcl
    acl_op_compiler
    securec
    pthread
    mockcpp
)

ADD_LIBRARY(ascendfaiss SHARED ${ASCENDFAISS_SRC})
target_include_directories(ascendfaiss PRIVATE ${ASCENDFAISS_INCLUDE})
target_link_directories(ascendfaiss PRIVATE ${ASCENDFAISS_LIBRARY_PATH})
TARGET_LINK_LIBRARIES(ascendfaiss PRIVATE ${ASCENDFAISS_LIBRARY})
INSTALL(TARGETS  ascendfaiss  DESTINATION  lib)

SET(TEST_SRC_FILES TestAscendIndexUTBinaryFlat.cpp
                   TestAscendIndexILFlatUT.cpp
                   TestAscendIndexUTCluster.cpp
                   TestAscendIndexUTClustering.cpp
                   TestAscendIndexUTFlatAT.cpp
                   TestAscendIndexUTILFlat.cpp
                   TestAscendIndexUTInt8.cpp
                   TestAscendIndexUTIVFSQ.cpp
                   TestAscendIndexUTIVFSQT.cpp
                   TestAscendIndexUTSQ.cpp
                   TestAscendIndexUTTS.cpp)
SET(TARGET_OUT TestAscendIndexUT)
ADD_EXECUTABLE(${TARGET_OUT}  ${TEST_SRC_FILES})

target_include_directories(${TARGET_OUT} PRIVATE ${ASCENDFAISS_INCLUDE})

target_link_options(${TARGET_OUT} PRIVATE 
    -Wl,--disable-new-dtags
)

target_link_directories(${TARGET_OUT} PRIVATE 
    ${ASCENDFAISS_LIBRARY_PATH}
)

TARGET_LINK_LIBRARIES(${TARGET_OUT} PRIVATE
    ascendfaiss
    gmock_main 
    gmock 
    gtest  
    ${ASCENDFAISS_LIBRARY}
)

IF(${ASAN_OPTION} STREQUAL "on")
    TARGET_LINK_LIBRARIES(${TARGET_OUT} PRIVATE
        -fsanitize=address
        -fsanitize=leak
        -fsanitize-recover=address,all
        -fno-omit-frame-pointer
    )
ENDIF()

SET_TARGET_PROPERTIES(${TARGET_OUT} PROPERTIES INSTALL_RPATH 
    "${GTEST_HOME}/lib;${CMAKE_INSTALL_PREFIX}/lib"
)

ADD_SUBDIRECTORY(cases)

INSTALL(TARGETS  ${TARGET_OUT}  DESTINATION  bin)


ADD_TEST(NAME ${TARGET_OUT}
        COMMAND ${TARGET_OUT} --gtest_output=xml
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})