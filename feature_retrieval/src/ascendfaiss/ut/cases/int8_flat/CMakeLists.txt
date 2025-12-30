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
 
project(TestAscendInt8Flat)
 
create_testable_target(${PROJECT_NAME}
    SOURCE_FILE
        TestAscendInt8Flat.cpp
 
    DEPENDS
        # to be tested
        ascendfaiss
 
        # 3rd lib
        gtest_main
        gtest
        ${ASCENDFAISS_LIBRARY}
 
    INCLUDES
        ${ASCENDFAISS_INCLUDE}
)