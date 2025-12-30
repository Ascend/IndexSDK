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

set(CMAKE_COMPILE ${CMAKE_CXX_COMPILER})

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/func.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/intf.cmake)

add_subdirectory(op_host)
add_subdirectory(op_kernel)

get_system_info(SYSTEM_INFO)

# gen version.info
add_custom_target(gen_version_info ALL
        COMMAND bash ${GEN_VERSION_FINO_SH} ${ASCEND_TOOLKIT_PATH} ${OPP_INSTALL_DIR}
)
