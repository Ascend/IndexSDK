cmake_minimum_required(VERSION 3.14.1)
set(CMAKE_CXX_STANDARD 14)

if (NOT DEFINED ASCEND_TOOLKIT_PATH)
    set(ASCEND_TOOLKIT_PATH "/usr/local/Ascend/ascend-toolkit/latest"   CACHE  STRING "")
endif()

if (NOT DEFINED ASCEND_HOME)
    set(ASCEND_HOME /usr/local/Ascend CACHE STRING "")
endif()

if (NOT DEFINED FAISS_HOME)
    set(FAISS_HOME /usr/local/ CACHE STRING "")
endif()

if (NOT DEFINED DRIVER_HOME)
    set(DRIVER_HOME /usr/local/Ascend CACHE STRING "")
endif()

if (NOT DEFINED CPU_TYPE)
    execute_process(COMMAND arch OUTPUT_VARIABLE CPU_TYPE OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if (NOT DEFINED HMM_ROOT_PATH)
    set(HMM_ROOT_PATH ${CMAKE_CURRENT_LIST_DIR}/../../..)
endif()

set(OUTPUT_PATH "${HMM_ROOT_PATH}/output" )
set(HMM_SRC_PATH "${HMM_ROOT_PATH}/src")
set(HMM_TEST_PATH "${HMM_ROOT_PATH}/tests")
set(HMM_INSTALL_PATH "${HMM_ROOT_PATH}/install")
set(HMM_THIRDPART_PATH ${HMM_ROOT_PATH}/3rdparty)
set(LIB_HW_SECURE ${HMM_THIRDPART_PATH}/huawei_secure_c/lib/libsecurec.so)
set(HMM_INTERFACE_ROOT_DIR ${HMM_ROOT_PATH}/interface)
set(HMM_INTRAFACE_ROOT_DIR ${HMM_ROOT_PATH}/intraface)
set(HCC ${ASCEND_TOOLKIT_PATH}/toolkit/toolchain/hcc)
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

add_compile_options(
        -Wreturn-type
        -Wno-unused-parameter
        -Wno-unused-function
        -Wunused-variable
        -Wunused-value
        -Wcast-align
        -Wcast-qual
        -Winvalid-pch
        -Wwrite-strings
        -Wsign-compare
        -Wfloat-equal
        -Wall
        -Wextra
        -std=c++14
        -DUSE_ACL_INTERFACE_V2)

add_compile_options(
        -Weffc++
        -Werror
        -Wtrampolines
        -Wformat=2
        -Wdate-time
        -Wswitch-default
        -Wshadow
        -Wconversion
        -Wvla
        -Wunused
        -Wundef
        -Wnon-virtual-dtor
        -Wdelete-non-virtual-dtor
        -Woverloaded-virtual)

add_compile_options(
        -fno-common
        -freg-struct-return
        -fstrong-eval-order)

add_compile_options(-pipe)

add_link_options(-rdynamic)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
add_link_options(-Wl,-z,relro,-z,now,-z,noexecstack)
    add_compile_options(
            -g
            -O0)
else()
    add_link_options(-Wl,-z,relro,-z,now,-z,noexecstack,-s)
    add_compile_options(
            -fstack-protector-all
            -fstack-protector-strong
            -fno-strict-aliasing
            -frename-registers
            -fopenmp
            -fms-extensions
            -D_FORTIFY_SOURCE=2
            -O3
            -s)
endif()

if (USING_COVERAGE STREQUAL "ON")
    add_compile_options(--coverage)
    add_link_options(--coverage)
endif()

if(USING_XSAN STREQUAL "ON")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

include_directories(
    ${HMM_INTERFACE_ROOT_DIR}
    ${HMM_INTRAFACE_ROOT_DIR}
    ${HMM_SRC_PATH}
    ${ASCEND_TOOLKIT_PATH}/include
    ${HMM_THIRDPART_PATH}/huawei_secure_c/include)

link_directories(
    ${OUTPUT_PATH}/memory_bridge/lib
    ${HMM_THIRDPART_PATH}/huawei_secure_c/lib
    ${FAISS_HOME}/lib
    ${ASCEND_TOOLKIT_PATH}/lib64
)
include(${CMAKE_CURRENT_LIST_DIR}/${CPU_TYPE}_tool_chain.cmake)
set(CMAKE_VERBOSE_MAKEFILE ON)
