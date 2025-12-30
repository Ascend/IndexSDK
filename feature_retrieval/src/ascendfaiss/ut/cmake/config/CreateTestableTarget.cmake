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

cmake_minimum_required(VERSION 3.20)

# 入参说明：
# TARGET_NAME 生成的可执行文件名称；
# SOURCE_FILE 测试源码文件；
# DEPENDS 依赖的其他target；
# INCLUDES 头文件引用路径；
function(create_testable_target TARGET_NAME)
    set(MULTI_VALUE_ARGS SOURCE_FILE DEPENDS INCLUDES)
    cmake_parse_arguments(FUNC "" "" "${MULTI_VALUE_ARGS}" ${ARGN})

    set(CUR_OBJECT_LIBRARY ${TARGET_NAME}-object)
    add_library(${CUR_OBJECT_LIBRARY} OBJECT
        ${FUNC_SOURCE_FILE}
    )

    target_include_directories(${CUR_OBJECT_LIBRARY} PRIVATE ${FUNC_INCLUDES})

    # 链接至最外层可执行文件，确保门禁覆盖率检测
    target_link_libraries(${TARGET_OUT} PRIVATE
        ${CUR_OBJECT_LIBRARY}
    )

    add_executable(${TARGET_NAME}
        $<TARGET_OBJECTS:${CUR_OBJECT_LIBRARY}>
    )

    # 后续优化为导入gtest target，链接路径导入target里，这里先临时写死
    target_link_directories(${TARGET_NAME} PRIVATE
        ${ASCENDFAISS_LIBRARY_PATH}
    )

    target_link_libraries(${TARGET_NAME} PRIVATE
        ${FUNC_DEPENDS}
        gmock
    )

    if(${ASAN_OPTION} STREQUAL "on")
        target_link_libraries(${TARGET_NAME} PRIVATE
            -fsanitize=address
            -fsanitize=leak
            -fsanitize-recover=address,all
            -fno-omit-frame-pointer
        )
    endif()

    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
endfunction()