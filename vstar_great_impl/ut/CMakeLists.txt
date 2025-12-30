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

cmake_minimum_required(VERSION 3.16.0)
project(IndexUT)

OPTION(ASAN_OPTION "ASAN SWITCH" OFF)
OPTION(USE_LOCAL_OPTION "USE_LOCAL SWITCH" OFF)

set(TARGET_LIBRARY_VSTAR vstar)

set(PROJECT_SRC_ROOT_VSTAR ${CMAKE_CURRENT_LIST_DIR}/../mix-index/)

IF (NOT DEFINED FAISS_HOME)
    SET(FAISS_HOME /usr/local/faiss/faiss1.10.0/ CACHE  STRING "") # todo: need to change this to CI's faiss location
ENDIF()

set(LIBRARY_OUTPUT_PATH ${PROJECT_SRC_ROOT_VSTAR}/lib)
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

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

# make sure OMP package is used
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS          "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OPENMP NOT FOUND")
endif()

# UT related compile flags
SET(GCC_COVERAGE_COMPILE_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
SET(GCC_COVERAGE_LINK_FLAGS "-lgcov")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")

add_compile_definitions(USE_ACL_INTERFACE_V2=1)
add_compile_definitions(HOSTCPU)
add_compile_definitions(INFO)
add_compile_options(
    -std=c++17
    -falign-jumps=64
)
add_link_options(-Wl,-z,relro -Wl,-z,now  -Wl,-z,noexecstack -s)

# Set file path

set (AMODE_DIR ${PROJECT_SRC_ROOT_VSTAR})
set (KMODE_DIR ${PROJECT_SRC_ROOT_VSTAR}/third_party/)

# define AMode's source file location, decide building in ACLOnlineMode vs ACLOfflineMode
FILE(GLOB_RECURSE AMODE_SRC_FILES ${AMODE_DIR}/src/*.cpp)

# define KMode's source file location
file(GLOB KMODE_SRC_FILES
    ${KMODE_DIR}/src/*.cpp
    ${KMODE_DIR}/src/impl/*.cpp
    ${KMODE_DIR}/src/utils/*.cpp
)

add_library(${TARGET_LIBRARY_VSTAR} SHARED ${KMODE_SRC_FILES} ${AMODE_SRC_FILES})

target_include_directories(${TARGET_LIBRARY_VSTAR} PRIVATE

    ${KMODE_DIR}/include
    ${AMODE_DIR}/include
    ${FAISS_HOME}/include

    # UT-related includes
    ${ACLMOCK_DIR}/
    ${ACLMOCK_DIR}/acl
    ${MOCKCPP_DIR}/output/include
    ${SECUREC_DIR}/include
    ${GTEST_DIR}/include
)

target_link_directories(${TARGET_LIBRARY_VSTAR} PRIVATE

    ${FAISS_HOME}/lib64
    ${FAISS_HOME}/lib

    # UT-related links
    ${ACLMOCK_DIR}/acl/lib
    ${MOCKCPP_DIR}/output/lib
    ${SECUREC_DIR}/lib
    ${GTEST_DIR}/lib64
    ${GTEST_DIR}/lib
)

target_link_libraries(${TARGET_LIBRARY_VSTAR} PRIVATE
    faiss
    ascendcl
    securec
    pthread
    mockcpp
)

INSTALL(TARGETS ${TARGET_LIBRARY_VSTAR} DESTINATION lib)


#######################################################################

# Step 2: Compile Unit Test exeuctables

SET(TEST_SRC_FILES
    TestVstar.cpp
    TestGreat.cpp
)
SET(UT_TARGET_OUT TestAscendIndexUT)
ADD_EXECUTABLE(${UT_TARGET_OUT}  ${TEST_SRC_FILES})

target_link_directories(${UT_TARGET_OUT} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/output/lib

    # UT-related links
    ${MOCKCPP_DIR}/output/lib
    ${GTEST_DIR}/lib64
    ${GTEST_DIR}/lib
)

target_link_options(${UT_TARGET_OUT} PRIVATE 
    -Wl,--disable-new-dtags
)

TARGET_INCLUDE_DIRECTORIES(${UT_TARGET_OUT} PRIVATE
    ${KMODE_DIR}/include
    ${AMODE_DIR}/include
    ${FAISS_HOME}/include

    # UT-related includes
    ${ACLMOCK_DIR}/
    ${ACLMOCK_DIR}/acl
    ${MOCKCPP_DIR}/output/include
    ${SECUREC_DIR}/include
    ${GTEST_DIR}/include
)

TARGET_LINK_LIBRARIES(${UT_TARGET_OUT} PRIVATE
    vstar
    gmock_main
    gmock
    gtest
    mockcpp
)

IF(${ASAN_OPTION} STREQUAL "on")
    TARGET_LINK_LIBRARIES(${UT_TARGET_OUT} PRIVATE
        -fsanitize=address
        -fsanitize=leak
        -fsanitize-recover=address,all
        -fno-omit-frame-pointer
    )
ENDIF()

ENABLE_TESTING()

INSTALL(TARGETS ${UT_TARGET_OUT} DESTINATION test)

ADD_TEST(NAME ${UT_TARGET_OUT}
        COMMAND ${UT_TARGET_OUT} --gtest_output=xml
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})