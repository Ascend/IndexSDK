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

# Step 1: Compile ascendsearch library which includes mock libraries

CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
PROJECT(ascendsearchUT)

OPTION(ASAN_OPTION "ASAN SWITCH" OFF)
OPTION(USE_LOCAL_OPTION "USE_LOCAL SWITCH" OFF)

INCLUDE(${CMAKE_CURRENT_LIST_DIR}/../cmake/utils.cmake)

ENABLE_TESTING()

IF (NOT DEFINED FAISS_HOME)
    SET(FAISS_HOME /usr/local/faiss/faiss1.10.0/ CACHE  STRING "") # todo: need to change this to CI's faiss location
ENDIF()

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_SKIP_BUILD_RPATH TRUE)

ADD_COMPILE_OPTIONS(
        -fPIC
        -fstack-protector-all
        -Wall
        -fno-strict-aliasing
        -Wreturn-type
        -frename-registers
        -fpeel-loops
        -fopenmp
        -fno-access-control # Allow user to access private/protected variables for testing purposes
        -Wno-pmf-conversions # this flag suppresses warnings for implicit conversions between pointer-to-member-function (PMF) types
)

IF(${ASAN_OPTION} STREQUAL "on")
    ADD_COMPILE_OPTIONS(
        -fsanitize=address
        -fsanitize=leak
        -fsanitize-recover=address,all
        -fno-omit-frame-pointer
    )
ENDIF()

# UT related compile flags
SET(GCC_COVERAGE_COMPILE_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
SET(GCC_COVERAGE_LINK_FLAGS "-lgcov")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

ADD_DEFINITIONS(
    -DUSE_ACL_INTERFACE_V2
    -DHOSTCPU
)

ADD_LINK_OPTIONS(
    -Wl,-z,relro  -Wl,-z,now  -Wl,-z,noexecstack  -s
)

SET(TARGET_LIBRARY         ascendsearch)
SET(RETRIEVAL_COMMON_DIR   ../../../Retrieval_Common)
SET(ASCENDHOST_DIR ${CMAKE_CURRENT_LIST_DIR}/../ascendhost)

SET(ASCENDSEARCH_SRC
    ${RETRIEVAL_COMMON_DIR}/ascend/AscendIndex.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/AscendMultiIndexSearch.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/AscendIndexIVFSPSQ.cpp
    ${RETRIEVAL_COMMON_DIR}/ascend/impl/AscendIndexImpl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascend/custom/impl/AscendIndexIVFSPSQImpl.cpp
    ${RETRIEVAL_COMMON_DIR}/ascend/utils/fp16.cpp
    ${RETRIEVAL_COMMON_DIR}/ascend/utils/MergeResUtils.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/StandardAscendResources.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/impl/AuxIndexStructures.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/impl/Index.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/impl/IndexIVF.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl_custom/IndexIVFSPSQ.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/impl_custom/IndexIVFSPSQL2.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/AscendMemory.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/AscendOpDesc.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/AscendOperator.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/AscendStackMemory.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/AscendUtils.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/MemorySpace.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/TopkOp.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/index_custom/IndexIVFSPSQL2Aicpu.cpp
    ${RETRIEVAL_COMMON_DIR}/ascendhost/src/rpc-local/AscendRpcLocal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascendhost/src/rpc-local/AscendRpcLocalIndexIVFSPSQ.cpp
    ${RETRIEVAL_COMMON_DIR}/common/threadpool/AscendThreadPool.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/IoUtil.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon/IVFSPCodeBookTrainer.cpp
)

SET(ASCENDSEARCH_INCLUDE
    ${CMAKE_CURRENT_LIST_DIR}/../
    ${RETRIEVAL_COMMON_DIR}/
    ${CMAKE_CURRENT_LIST_DIR}/../ascend
    ${RETRIEVAL_COMMON_DIR}/ascend
    ${RETRIEVAL_COMMON_DIR}/ascend/utils
    ${CMAKE_CURRENT_LIST_DIR}/../ascenddaemon
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/impl
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils
    ${RETRIEVAL_COMMON_DIR}/common
    ${ASCENDHOST_DIR}/include
    ${RETRIEVAL_COMMON_DIR}/ascendhost/include
    ${ASCENDHOST_DIR}/include/impl
    ${RETRIEVAL_COMMON_DIR}/ascendhost/include/impl
    ${ASCENDHOST_DIR}/include/index
    ${RETRIEVAL_COMMON_DIR}/ascendhost/include/index
    ${FAISS_HOME}/include
    ${ASCEND_HOME}/driver/include/dvpp
    "/usr/local/include/"
    /opt/OpenBLAS/include/

    # UT-related includes
    ${ACLMOCK_DIR}/
    ${ACLMOCK_DIR}/acl
    ${MOCKCPP_DIR}/output/include
    ${SECUREC_DIR}/include
    ${GTEST_DIR}/include
)

SET(ASCENDSEARCH_LIBRARY_PATH
    ${FAISS_HOME}/lib64

    # UT-related links
    ${ACLMOCK_DIR}/acl/lib
    ${MOCKCPP_DIR}/output/lib
    ${SECUREC_DIR}/lib
    ${GTEST_DIR}/lib64
)

if(${USE_LOCAL_OPTION} STREQUAL "on")
    list(APPEND ASCENDSEARCH_LIBRARY_PATH
        ${FAISS_HOME}/lib64
        ${GTEST_DIR}/lib64
    )
else()
    list(APPEND ASCENDSEARCH_LIBRARY_PATH
        ${FAISS_HOME}/lib
        ${GTEST_DIR}/lib
    )
endif()

ADD_LIBRARY(${TARGET_LIBRARY}  SHARED  ${ASCENDSEARCH_SRC})
TARGET_INCLUDE_DIRECTORIES(${TARGET_LIBRARY} PRIVATE ${ASCENDSEARCH_INCLUDE})
TARGET_LINK_DIRECTORIES(${TARGET_LIBRARY} PRIVATE ${ASCENDSEARCH_LIBRARY_PATH})

GET_FILENAME_COMPONENT(ASCENDFAISS_DIR   ${CMAKE_CURRENT_LIST_DIR}/..  ABSOLUTE)
redefine_file_macro(${TARGET_LIBRARY}  ${ASCENDFAISS_DIR})
TARGET_LINK_LIBRARIES(${TARGET_LIBRARY} PRIVATE 
    faiss
    ascendcl
    securec
    pthread
    mockcpp
)
INSTALL(TARGETS ${TARGET_LIBRARY} DESTINATION lib)

#######################################################################

# Step 2: Compile Unit Test exeuctables

SET(TEST_SRC_FILES
    TestAscendIndexIVFSP_NormalCases.cpp
    TestAscendIndexIVFSP_EdgeCases.cpp
)
SET(TARGET_OUT TestAscendIndexUT)
ADD_EXECUTABLE(${TARGET_OUT}  ${TEST_SRC_FILES})

target_include_directories(${TARGET_OUT} PRIVATE ${ASCENDSEARCH_INCLUDE})

target_link_options(${TARGET_OUT} PRIVATE 
    -Wl,--disable-new-dtags
)

target_link_directories(${TARGET_OUT} PRIVATE
    ${ASCENDSEARCH_LIBRARY_PATH}
    ${CMAKE_CURRENT_LIST_DIR}/output/lib
)

TARGET_LINK_LIBRARIES(${TARGET_OUT} PRIVATE
    ascendsearch
    gmock_main
    gmock
    gtest
    faiss
    ascendcl
    securec
    pthread
    mockcpp
)

IF(${ASAN_OPTION} STREQUAL "on")
    TARGET_LINK_LIBRARIES(${TARGET_OUT} PRIVATE
        -fsanitize=address
        -fsanitize=leak
        -fsanitize-recover=address,all
        -fno-omit-frame-pointer
    )
ENDIF()

INSTALL(TARGETS ${TARGET_OUT} DESTINATION test)

ADD_TEST(NAME ${TARGET_OUT}
        COMMAND ${TARGET_OUT} --gtest_output=xml
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})