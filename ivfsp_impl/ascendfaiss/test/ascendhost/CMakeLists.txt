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
set(TESTCASE_SRC_DIR "${PROJECT_SOURCE_DIR}/../ascendhost")
SET(RETRIEVAL_COMMON_DIR   ../../../../Retrieval_Common)
# for objects from original feature retrieval open form
set(ASCENDFAISS_HOST_SRC
    "${MXINDEX_SRC_ROOT_DIR}/ascend/custom/AscendIndexIVFSPSQ.cpp"
)

set(ASCENDFAISS_TEST_SRC
    "${TESTCASE_SRC_DIR}/TestAscendIndexIVFSPSQ.cpp"
)

set(ASCENDFAISS_HOST_INC_DIR
    "${RETRIEVAL_COMMON_DIR}/"
    "${MXINDEX_SRC_ROOT_DIR}/ascendhost/include"
    "${RETRIEVAL_COMMON_DIR}/ascendhost/include"
    "${MXINDEX_SRC_ROOT_DIR}"
    "${MXINDEX_SRC_ROOT_DIR}/ascend"
    "${RETRIEVAL_COMMON_DIR}/ascend"
    "${RETRIEVAL_COMMON_DIR}/ascend/utils"
    "${MXINDEX_SRC_ROOT_DIR}/ascenddaemon"
    "${RETRIEVAL_COMMON_DIR}/ascenddaemon"
    "${RETRIEVAL_COMMON_DIR}/ascenddaemon/impl"
    "${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils"
    "${RETRIEVAL_COMMON_DIR}/common"

    "${ASCEND_HOME}/ascend-toolkit/latest/acllib/include"
    "${ASCEND_HOME}/driver/kernel/libc_sec/include"
    "${ASCEND_HOME}/driver/include/dvpp"
    "/usr/local/include"
#        "${MXINDEX_SRC_ROOT_DIR}/test/ascendhost/gtest/include"
    "/usr/local/gtest/include"
    "/usr/local/protobuf/include"
    "/usr/local/Ascend/driver/kernel/inc/driver"
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
    "/usr/local/Ascend/driver/lib64/common/"
    "/usr/local/gtest/lib64/"
    "/usr/local/Ascend/ascend-toolkit/5.1.RC2/opp/op_impl/built-in/ai_core/tbe/op_tiling/"
	"/home/z00604525/20230709/Retrieval/ascendfaiss/build/ascendhost"
)

target_link_libraries(test_ascendhost PRIVATE
    ascendsearch
    faiss
    ascendcl
    c_sec
    pthread
    gomp
    gtest
    gtest_main
)

target_compile_options(test_ascendhost PRIVATE
    -g
    -O0
    -std=c++14
    -DHOSTCPU
    -DDEBUG
#    -D_GLIBCXX_USE_CXX11_ABI=0
)

#add_subdirectory(../../../third_party/googletest/googletest googletest)
