cmake_minimum_required(VERSION 3.13)

project(image_duplicate_retrieval LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(INDEXSDK_ROOT "/usr/local/Ascend/mxIndex" CACHE PATH "IndexSDK install root")
set(FAISS_ROOT "/usr/local/faiss" CACHE PATH "Faiss install root")
set(CANN_ROOT "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "CANN toolkit root")
set(ASCEND_DRIVER_ROOT "/usr/local/Ascend/driver" CACHE PATH "Ascend driver root")
set(OPENBLAS_ROOT "/opt/OpenBLAS" CACHE PATH "OpenBLAS install root")

if(NOT EXISTS "${FAISS_ROOT}/include/faiss/IndexIVFRaBitQ.h")
    message(FATAL_ERROR "Faiss 1.14 or newer with IndexIVFRaBitQ is required: ${FAISS_ROOT}")
endif()
if(NOT EXISTS "${INDEXSDK_ROOT}/include/faiss/ascend/AscendIndexIVFRaBitQ.h")
    message(FATAL_ERROR "IndexSDK with AscendIndexIVFRaBitQ is required: ${INDEXSDK_ROOT}")
endif()

find_package(OpenBLAS REQUIRED)

add_library(indexsdk_demo_dependencies INTERFACE)
target_include_directories(indexsdk_demo_dependencies INTERFACE
    ${INDEXSDK_ROOT}/include
    ${INDEXSDK_ROOT}/include/faiss
    ${FAISS_ROOT}/include
    ${CANN_ROOT}/include
    ${ASCEND_DRIVER_ROOT}/include
    ${OPENBLAS_ROOT}/include
)
target_link_directories(indexsdk_demo_dependencies INTERFACE
    ${INDEXSDK_ROOT}/host/lib
    ${ASCEND_DRIVER_ROOT}/lib64
    ${ASCEND_DRIVER_ROOT}/lib64/driver
    ${FAISS_ROOT}/lib64
    ${FAISS_ROOT}/lib
    ${CANN_ROOT}/lib64
    ${OPENBLAS_ROOT}/lib
)
target_link_options(indexsdk_demo_dependencies INTERFACE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    -pie
)
target_link_libraries(indexsdk_demo_dependencies INTERFACE
    ascendfaiss
    faiss
    openblas
    c_sec
    ascendcl
    ascend_hal
    acl_op_compiler
    pthread
)

function(add_indexsdk_demo target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE indexsdk_demo_dependencies)
    set_target_properties(${target} PROPERTIES BUILD_RPATH
        "${INDEXSDK_ROOT}/host/lib;${FAISS_ROOT}/lib64;${FAISS_ROOT}/lib;${CANN_ROOT}/lib64;${ASCEND_DRIVER_ROOT}/lib64;${ASCEND_DRIVER_ROOT}/lib64/driver;${OPENBLAS_ROOT}/lib")
endfunction()

add_indexsdk_demo(image_dedup_business src/main.cpp)
add_indexsdk_demo(image_dedup_benchmark src/benchmark.cpp)
