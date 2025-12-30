cmake_minimum_required(VERSION 3.5.2)
project(ascendhost_test)

# Skip build rpath
set(CMAKE_SKIP_BUILD_RPATH True)

option(BUILD_SAMPLE "Enable build sample." OFF)

# check environment variable
set(ASCEND_HOME /usr/local/Ascend)
if(DEFINED ENV{ASCEND_HOME})
    set(ASCEND_HOME $ENV{ASCEND_HOME})
endif()

set(DRIVER_HOME ${ASCEND_HOME})
if(DEFINED ENV{DRIVER_HOME})
    set(DRIVER_HOME $ENV{DRIVER_HOME})
endif()

set(MXINDEX_SRC_ROOT_DIR "${PROJECT_SOURCE_DIR}/../..")
set(TESTCASE_SRC_DIR "${PROJECT_SOURCE_DIR}/../ascend")

# for objects from original feature retrieval open form
set(ASCENDFAISS_HOST_SRC
    "${MXINDEX_SRC_ROOT_DIR}/ascend/AscendIndex.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascend/AscendIndexFlat.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascend/AscendIndexIVF.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascend/AscendIndexIVFSQ.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascend/custom/AscendClustering.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascend/custom/AscendIndexFlatAT.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascend/custom/AscendIndexFlatATInt8.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascend/utils/fp16.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/StandardAscendResources.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/ThreadSafeStandardAscendResources.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/impl/AuxIndexStructures.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/impl/Index.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/impl/IndexFlat.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/impl/IndexFlatIP.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/impl/IndexIVF.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/impl/IndexIVFSQ.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/utils/AscendMemory.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/utils/AscendOpDesc.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/utils/AscendOperator.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/utils/AscendStackMemory.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/utils/AscendUtils.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/utils/AscendRWLock.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/utils/DeviceMemMng.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/utils/ThreadSafeDistComputeOpsManager.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/utils/DistComputeOpsManager.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/utils/MemorySpace.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/utils/TopkOp.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascendhost/src/index/IndexFlatIPAicpu.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/ascendhost/src/index/IndexIVFSQIPAicpu.cpp"
    "${MXINDEX_SRC_ROOT_DIR}/common/threadpool/AscendThreadPool.cpp"
)

set(ASCENDFAISS_TEST_SRC
    #"${TESTCASE_SRC_DIR}/TestAscendIndexFlat.cpp"
    "${TESTCASE_SRC_DIR}/TestAscendIndexIVFSQ.cpp"
)

set(ASCENDFAISS_HOST_INC_DIR
    "${MXINDEX_SRC_ROOT_DIR}/ascendhost/include"
    "${MXINDEX_SRC_ROOT_DIR}"
    "${MXINDEX_SRC_ROOT_DIR}/ascend"
    "${MXINDEX_SRC_ROOT_DIR}/ascend/utils"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/impl"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon/utils"
    "${MXINDEX_SRC_ROOT_DIR}/common"
    "${ASCEND_HOME}/ascend-toolkit/latest/acllib/include"
    "${ASCEND_HOME}/driver/kernel/libc_sec/include"
    "${ASCEND_HOME}/driver/include/dvpp"
    "/usr/local/include"
)

add_executable(test_ascendhost
    ${ASCENDFAISS_HOST_SRC}
    ${ASCENDFAISS_TEST_SRC}
)

target_include_directories(test_ascendhost PRIVATE
    ${ASCENDFAISS_HOST_INC_DIR}
)

target_link_directories(test_ascendhost PRIVATE
    ${ASCEND_HOME}/driver/lib64
    ${ASCEND_HOME}/ascend-toolkit/latest/acllib/lib64
    "/usr/local/lib"
)

target_link_libraries(test_ascendhost PRIVATE
    faiss
    ascendcl
    c_sec
    gtest
    gtest_main
    pthread
    gomp
)

target_compile_options(test_ascendhost PRIVATE
    -g
    -O0
    -std=c++14
    -DHOSTCPU
    -DDEBUG
)

add_subdirectory(../../third_party/googletest/googletest googletest)