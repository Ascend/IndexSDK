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
SET(TARGET_NAME  memory_bridge_test)
project(${TARGET_NAME} VERSION 1.0 LANGUAGES C CXX)

# 包含所有组件的cmake
include("${CMAKE_CURRENT_SOURCE_DIR}/conf/comp.cmake")

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/../install/securec/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../interface"
    "${CMAKE_CURRENT_SOURCE_DIR}/../intraface"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    "${CMAKE_CURRENT_SOURCE_DIR}/mock"
    "${CMAKE_CURRENT_SOURCE_DIR}/env"
    "${ASCEND_TOOLKIT_PATH}/include"
)
link_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/../output/memory_bridge/lib"
    "${CMAKE_CURRENT_SOURCE_DIR}/../install/securec/lib"
    "${ASCEND_TOOLKIT_PATH}/lib64"
)

# 添加测试源代码
aux_source_directory(ock/hmm/mgr TEST_SRC)
aux_source_directory(ock/hmm/mgr/checker TEST_SRC)
aux_source_directory(ock/hmm/mgr/data TEST_SRC)
aux_source_directory(ock/hmm/mgr/api TEST_SRC)
aux_source_directory(ock/hmm/mgr/algo TEST_SRC)
aux_source_directory(ock/hmm/helper TEST_SRC)
aux_source_directory(ock/log TEST_SRC)
aux_source_directory(env/ock TEST_SRC)
aux_source_directory(env/ock/acladapter TEST_SRC)
aux_source_directory(ock/acladapter/executor TEST_SRC)
aux_source_directory(ock/acladapter/param TEST_SRC)
aux_source_directory(ock/acladapter/task TEST_SRC)
aux_source_directory(ock/acladapter/utils TEST_SRC)
aux_source_directory(ock/acladapter/data TEST_SRC)
aux_source_directory(ock/tools/topo TEST_SRC)
aux_source_directory(ock/utils TEST_SRC)
aux_source_directory(ock/conf TEST_SRC)
aux_source_directory(ock TEST_SRC)
aux_source_directory(fuzz/ock/acladapter/task TEST_SRC)
aux_source_directory(fuzz/ock/interface TEST_SRC)
aux_source_directory(performance/ock/hmm/mgr TEST_SRC)


# 生成目标
add_executable(${TARGET_NAME} ${TEST_SRC})

# 打印构建选项
get_target_property(COMPILE_FLAGS ${TARGET_NAME} COMPILE_OPTIONS)
get_target_property(LINK_FLAGS ${TARGET_NAME} LINK_OPTIONS)
target_link_libraries(${TARGET_NAME} ock_hmm TopoDetectImpl securec ascendcl)
message(STATUS "Compiler id: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Compile flags: ${COMPILE_FLAGS}")
message(STATUS "Link flags: ${LINK_FLAGS}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")