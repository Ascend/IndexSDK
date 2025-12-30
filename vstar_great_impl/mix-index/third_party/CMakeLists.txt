cmake_minimum_required(VERSION 3.5.1)
project(AscendSearch)

set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(TARGET_LIBRARY graphsearch)
set(TARGET_LIBRARY_STATIC graphsearch_static)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SRC_ROOT}/lib)

set(CMAKE_CXX_STANDARD 11)

# disable ut test default
option(ENABLE_TEST "build test cases" OFF)

# using cmake .. -DUSE_NPU=ON/OFF to switch
option(USE_NPU "use npu to accelerate" ON)
if (USE_NPU)
    message(STATUS "Use npu for accelerating")
    include(cmake/NpuConfig.cmake)
else()
    message(STATUS "Do not use npu")
endif()

# change it if necessary
set(SECURE_API /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/)
message(STATUS "SECURE_API_INCLUDE_DIRS = ${SECURE_API}/include")
message(STATUS "SECURE_API_LIBS = ${SECURE_API}/lib64")

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OPENMP NOT FOUND")
endif()

# delete the HOSTCPU
add_compile_options(-std=c++11 -O3 -DINFO -fPIE -fstack-protector-all -fPIC -DHOSTCPU -Wall -Wextra -Wfloat-equal -w)

SET(PROCESS_TYPE ${CMAKE_HOST_SYSTEM_PROCESSOR})
MESSAGE(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR=${CMAKE_HOST_SYSTEM_PROCESSOR}")
IF (${PROCESS_TYPE} STREQUAL "x86_64")
    ADD_COMPILE_OPTIONS(-msse4.1)
ENDIF ()

set(CPU_SRC_DIR ${PROJECT_SRC_ROOT}/src)
set(NPU_SRC_DIR ${PROJECT_SRC_ROOT}/src/npu)
set(NPU_UTILS_DIR ${PROJECT_SRC_ROOT}/include)
set(OPS_UTILS_DIR ${PROJECT_SRC_ROOT}/ops/cpukernel/impl/utils)

if (USE_NPU)
    add_compile_options(-DUSE_NPU)
    # message(STATUS "ascend version = ${ASCEND_VERSION}")
    # message(STATUS "ascend home = ${ASCEND_HOME}")

    file(GLOB_RECURSE SRC_FILES
            ${CPU_SRC_DIR}/*.cpp
            ${CPU_SRC_DIR}/impl/*.cpp
            ${CPU_SRC_DIR}/utils/*.cpp

            ${NPU_SRC_DIR}/*.cpp
            ${NPU_SRC_DIR}/impl/*.cpp
            ${NPU_UTILS_DIR}/*.cpp
    )

    include_directories(
            ${SECURE_API}/include
            ${PROJECT_SRC_ROOT}/include
            ${NPU_UTILS_DIR}
            ${NPU_UTILS_DIR}/npu_utils
            ${OPS_UTILS_DIR}
            ${ASCEND_VERSION}/include
            ${ASCEND_HOME}/driver/include/dvpp
            ${ASCEND_HOME}/ascend-toolkit/latest/acllib/include
    )

    link_directories(
            ${SECURE_API}/lib64
            ${ASCEND_HOME}/ascend-toolkit/latest/acllib/lib64/
            ${ASCEND_VERSION}/lib64
    )

else()
    file(GLOB SRC_FILES
            ${CPU_SRC_DIR}/*.cpp
            ${CPU_SRC_DIR}/impl/*.cpp
            ${CPU_SRC_DIR}/utils/*.cpp
    )

    include_directories(
            ${SECURE_API}/include
            ${PROJECT_SRC_ROOT}/include
    )

    link_directories(
            ${SECURE_API}/lib64
    )
endif()

add_library(${TARGET_LIBRARY} SHARED ${SRC_FILES})

add_library(${TARGET_LIBRARY_STATIC} STATIC ${SRC_FILES})
set_target_properties(${TARGET_LIBRARY_STATIC} PROPERTIES OUTPUT_NAME ${TARGET_LIBRARY})

if (USE_NPU)
    target_link_libraries(${TARGET_LIBRARY} acl_retr ascendcl)
endif()

# 指定securec的搜索路径
set(LIBRARY_PATH /usr/local/Ascend/driver/lib64/common/)
find_library(SECUREC_LIBRARY NAMES securec PATHS ${LIBRARY_PATH} NO_DEFAULT_PATH)
if (SECUREC_LIBRARY)
    message(STATUS "Found securec library: ${SECUREC_LIBRARY}")
else()
    message(FATAL_ERROR "Could not find securec library")
endif()

target_link_libraries(${TARGET_LIBRARY} ${SECUREC_LIBRARY} pthread)