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

cmake_minimum_required(VERSION 3.14.1)

if (NOT DEFINED ASCEND_TOOLKIT_PATH)
    set(ASCEND_TOOLKIT_PATH "/usr/local/Ascend/ascend-toolkit/latest"   CACHE  STRING "")
endif()
SET(TARGET_NAME  hcps_pier_test)
project(${TARGET_NAME} VERSION 1.0 LANGUAGES C CXX)

# 包含所有组件的cmake
include("${CMAKE_CURRENT_SOURCE_DIR}/conf/comp.cmake")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/../../install/securec/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../install/memory_bridge/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../install/memory_bridge/intraface"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../interface"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../intraface"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src"
        "${CMAKE_CURRENT_SOURCE_DIR}/env"
        "${ASCEND_TOOLKIT_PATH}/include"
)
link_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/../../output/hcps_pier/lib"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../install/securec/lib"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../install/memory_bridge/lib"
        "${ASCEND_TOOLKIT_PATH}/lib64"
)

# 添加测试源代码
aux_source_directory(ock TEST_SRC)
aux_source_directory(ock/vsa TEST_SRC)
aux_source_directory(ock/hcps/nop TEST_SRC)
aux_source_directory(ock/hcps/hcop TEST_SRC)
aux_source_directory(ock/vsa/neighbor/base TEST_SRC)
aux_source_directory(ock/vsa/neighbor/npu TEST_SRC)

# 生成目标
add_executable(${TARGET_NAME} ${TEST_SRC})

# 打印构建选项
get_target_property(COMPILE_FLAGS ${TARGET_NAME} COMPILE_OPTIONS)
get_target_property(LINK_FLAGS ${TARGET_NAME} LINK_OPTIONS)
target_link_libraries(${TARGET_NAME} ock_hmm ock_hcps_pier securec ascendcl)
message(STATUS "Compiler id: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compile flags: ${COMPILE_FLAGS}")
message(STATUS "Link flags: ${LINK_FLAGS}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")