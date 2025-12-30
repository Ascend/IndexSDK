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

# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
endif()

foreach(compute_unit ${ASCEND_COMPUTE_UNIT})

    # generate aic-${compute_unit}-ops-info.json
    add_ops_info_target(TARGET ops_info_gen_${compute_unit}
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/aic-${compute_unit}-ops-info.json
        OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
        INSTALL_DIR packages/vendors/${vendor_name}/op_impl/ai_core/tbe/config/${compute_unit}
    )

    # generate ascendc impl py once
    if (NOT TARGET ascendc_impl_gen_${compute_unit})
        add_ops_impl_target(TARGET ascendc_impl_gen_${compute_unit}
            OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
            OPS_BATCH ${BATCH_MODE_REPLAY_LIST}
            OPS_ITERATE ${ITERATOR_MODE_REPLAY_LIST}
            IMPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}
            OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbe
            INSTALL_DIR packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl
        )
    endif()
endforeach()
