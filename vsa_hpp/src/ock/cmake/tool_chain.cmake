cmake_minimum_required(VERSION 3.14.1)
set(CMAKE_CXX_STANDARD 11)
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
if (NOT DEFINED PROJECT_ROOT_PATH)
    set(PROJECT_ROOT_PATH ${CMAKE_CURRENT_LIST_DIR}/../../..)
endif()
set(OUTPUT_PATH "${PROJECT_ROOT_PATH}/output" )
set(PROJECT_SRC_PATH "${PROJECT_ROOT_PATH}/src")
set(PROJECT_TEST_PATH "${PROJECT_ROOT_PATH}/tests")
set(PROJECT_INSTALL_PATH "${PROJECT_ROOT_PATH}/install")
set(PROJECT_THIRDPART_PATH ${PROJECT_ROOT_PATH}/3rdparty)
set(LIB_HW_SECURE ${PROJECT_THIRDPART_PATH}/huawei_secure_c/lib/libsecurec.so)
set(PROJECT_INTERFACE_ROOT_DIR ${PROJECT_ROOT_PATH}/interface)
set(PROJECT_INTRAFACE_ROOT_DIR ${PROJECT_ROOT_PATH}/intraface)
set(HCC ${ASCEND_TOOLKIT_PATH}/toolkit/toolchain/hcc)
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

set(CMAKE_CXX_STANDARD 11)
add_compile_options(-Wreturn-type -Wno-unused-parameter -Wno-unused-function -Wunused-variable -Wunused-value 
    -Wcast-align -Wcast-qual -Winvalid-pch -Wwrite-strings -Wsign-compare -Wfloat-equal -Wall -Wextra -std=c++11
    -DUSE_ACL_INTERFACE_V2)

add_compile_options(-Weffc++ -Werror -Wtrampolines -Wformat=2 -Wdate-time -Wswitch-default -Wshadow -Wconversion
    -Wvla -Wunused -Wundef -Wnon-virtual-dtor -Wdelete-non-virtual-dtor -Woverloaded-virtual)

add_compile_options(-fno-common -freg-struct-return -fstrong-eval-order)

add_compile_options(-pipe)

add_link_options(-Wl,-z,relro,-z,now,-z,noexecstack)
add_link_options(-rdynamic)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g)
    add_compile_options(-O0)
else()
    add_link_options(-s)
    add_compile_options(-fPIC -fstack-protector-all -fstack-protector-strong -fno-strict-aliasing 
        -frename-registers -fopenmp -fms-extensions -D_FORTIFY_SOURCE=2 -O3 -s)
endif()
if (USING_COVERAGE STREQUAL "ON")
    add_compile_options(-ftest-coverage -fprofile-arcs -fdump-rtl-expand)
    add_compile_options(--coverage)
    add_link_options(--coverage)
    link_libraries(gcov)
endif()
if(USING_XSAN STREQUAL "ON")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

function(include_ock_component FILE_PATH COMPONENT_NAME)
    message("FILE_PATH: ${FILE_PATH}")
    message("COMPONENT_NAME: ${COMPONENT_NAME}")

    include_directories(${FILE_PATH}/${COMPONENT_NAME}/include
                        ${FILE_PATH}/${COMPONENT_NAME}/intraface)

    link_directories(${FILE_PATH}/${COMPONENT_NAME}/lib)
endfunction()

include_ock_component(${PROJECT_INSTALL_PATH} "hcps")
include_ock_component(${PROJECT_INSTALL_PATH} "memory_bridge")
include_directories(
    ${PROJECT_INTERFACE_ROOT_DIR}
    ${PROJECT_INTRAFACE_ROOT_DIR}
    ${PROJECT_SRC_PATH}
    ${ASCEND_TOOLKIT_PATH}/include
    ${ASCEND_HOME}/driver/include/dvpp
)

link_directories(
    ${OUTPUT_PATH}/vsa/lib
    ${FAISS_HOME}/lib
    ${ASCEND_TOOLKIT_PATH}/lib64
)
include(${CMAKE_CURRENT_LIST_DIR}/${CPU_TYPE}_tool_chain.cmake)
set(CMAKE_VERBOSE_MAKEFILE ON)
