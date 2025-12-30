cmake_minimum_required(VERSION 3.14)
project(cpu_kernels_llt)

# ATTENTION:
# float16_t testcase can only run on aarch64!!

if("${ASCEND_CUSTOM_PATH}" STREQUAL "")
    message(WARNING "ASCEND_CUSTOM_PATH was not set, use env var ASCEND_AICPU_PATH instead.")
    if ("$ENV{ASCEND_AICPU_PATH}" STREQUAL "")
        message(FATAL_ERROR "ASCEND_AICPU_PATH was not set, compile failed.")
        return()
    endif()
    set(ASCEND_CUSTOM_PATH $ENV{ASCEND_AICPU_PATH})
endif()

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -DAICPU_UTEST)
set(AICPU_OPP_ENV ${ASCEND_CUSTOM_PATH}/opp/built-in/op_impl/aicpu/aicpu_kernel)
set(OP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../ops/cpukernel/impl)
set(INDEX_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

set(OP_IMPL_FILES
    ${OP_PATH}/topk_flat_cpu_kernel.cpp
    ${OP_PATH}/topk_ivf_cpu_kernel.cpp
    ${OP_PATH}/topk_ivf_fp32_cpu_kernel.cpp
    ${OP_PATH}/topk_ivf_fuzzy_cpu_kernel.cpp
    ${OP_PATH}/topk_ivfsqt_l1_cpu_kernel.cpp
)
set(OP_UTIL_FILES
    ${OP_PATH}/utils/kernel_utils.cpp
)
set(TEST_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_main.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/test_topk_flat.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/test_topk_ivf.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/test_topk_ivf_fuzzy.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/test_topk_ivfsqt_l1.cc
)

message(STATUS "OP_IMPL_FILES=${OP_IMPL_FILES}")
set(_cpu_kernels_llt_files
    ${TEST_FILES}
    ${OP_IMPL_FILES}
    ${OP_UTIL_FILES}
)

link_directories(
    ${AICPU_OPP_ENV}/lib/aarch64
    ${AICPU_OPP_ENV}/lib/x86
    ${ASCEND_CUSTOM_PATH}/compiler/lib64
    /usr/local/Ascend/driver/lib64
)

add_executable(cpu_kernels_llt
    ${_cpu_kernels_llt_files}
)

target_include_directories(cpu_kernels_llt PRIVATE
    ${AICPU_OPP_ENV}/inc
    ${INDEX_ROOT}/ascenddaemon/utils
    ${OP_PATH}
    ${OP_PATH}/utils
    /usr/local/Ascend/driver/include/dvpp
    ${ASCEND_CUSTOM_PATH}/include/
)

target_link_libraries(cpu_kernels_llt
    gtest
    gcov
    pthread
    cpu_kernels_context
    aicpu_nodedef_builder
    ascend_protobuf
    ascendalog
    c_sec
    -ldl
)

target_compile_options(cpu_kernels_llt PUBLIC
    -g
    -O0
    -fprofile-arcs
    -ftest-coverage
    -w
    -std=c++11
    -fPIC
)

set_target_properties(cpu_kernels_llt PROPERTIES OUTPUT_NAME "aicpu_all_ut" RUNTIME_OUTPUT_DIRECTORY .)

set(GOOGLETEST_VERSION 1.11.0)
add_subdirectory(../../third_party/googletest/googletest googletest)
