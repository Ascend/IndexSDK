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

# 包括CPP源文件的所有路径
SET(ASCEND_CORE_DIRS
    ${ASCEND_PROJECT_HOME}/ascend
    ${ASCEND_PROJECT_HOME}/ascenddaemon
    ${ASCEND_PROJECT_HOME}/ascendhost/src
    ${ASCEND_PROJECT_HOME}/common
)

SET(ASCENDFAISS_SRC "")

# 收集目录中的源文件
FOREACH(DIR ${ASCEND_CORE_DIRS})
    IF(EXISTS ${DIR})
        FILE(GLOB_RECURSE DIR_SOURCES "${DIR}/*.cpp")
        LIST(APPEND ASCENDFAISS_SRC ${DIR_SOURCES})
    ENDIF()
ENDFOREACH()

LIST(APPEND ASCENDFAISS_SRC
    ${CMAKE_CURRENT_LIST_DIR}/stub/hmm/HmmMock.cpp
    # for ilflat
    ${CMAKE_CURRENT_LIST_DIR}/Common.cpp
    # for great
    ${CMAKE_CURRENT_LIST_DIR}/stub/great/GreatMock.cpp
    # vstar
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
    ${OPEN_SOURCE_HOME}/mockcpp/mockcpp/output/include
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
    ${OPEN_SOURCE_HOME}/mockcpp/mockcpp/output/lib
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

FILE(GLOB TEST_SRC_FILES "${CMAKE_CURRENT_LIST_DIR}/Test*.cpp")

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