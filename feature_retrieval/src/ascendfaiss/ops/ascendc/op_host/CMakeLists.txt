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

set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(ops_srcs
    ${CMAKE_CURRENT_LIST_DIR}/ascendc_dist_int8_flat_cos.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ascendc_l2_norm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/distance_flat_ip.cpp
    ${CMAKE_CURRENT_LIST_DIR}/distance_flat_l2.cpp
    ${CMAKE_CURRENT_LIST_DIR}/op_host_common.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ascendc_dist_int8_flat_l2.cpp
    ${CMAKE_CURRENT_LIST_DIR}/distance_flat_l2_mins_at_fp32.cpp
    ${CMAKE_CURRENT_LIST_DIR}/distance_ivf_flat_ip_fp32.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ascendc_distance_batch_mask_generator.cpp
    ${CMAKE_CURRENT_LIST_DIR}/ascendc_distance_batch_mask_generator_with_extra.cpp
)

set(ascend_search_ops_srcs
    ${CMAKE_CURRENT_LIST_DIR}/vstar_compute_l1.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vstar_compute_l2.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vsc3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vsm3.cpp
    ${CMAKE_CURRENT_LIST_DIR}/vstar_base_add_mat_mul.cpp
)

if (ASCEND_SEARCH)
    list(APPEND ops_srcs ${ascend_search_ops_srcs})
endif()

opbuild(OPS_SRC ${ops_srcs}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

add_library(cust_op_proto_ascendc SHARED ${ops_srcs} ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
target_compile_definitions(cust_op_proto_ascendc PRIVATE OP_PROTO_LIB)
target_compile_options(cust_op_proto_ascendc PRIVATE
        -fvisibility=hidden
)
if (ENABLE_CROSS_COMPILE)
    target_link_directories(cust_op_proto_ascendc PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_op_proto_ascendc PRIVATE
        intf_pub
        exe_graph
        register
        tiling_api
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_op_proto_ascendc PROPERTIES OUTPUT_NAME
                      cust_opsproto_rt2.0
)
add_library(cust_optiling SHARED ${ops_srcs})
target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
target_compile_options(cust_optiling PRIVATE
        -fvisibility=hidden
)
target_link_libraries(cust_optiling PRIVATE
        intf_pub
        exe_graph
        register
        tiling_api
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME
                      cust_opmaster_rt2.0
)

add_custom_target(optiling_compat ALL
                  COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_optiling>
                          ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
                  COMMAND mkdir -p ${OPP_INSTALL_DIR}/op_proto/inc
                  COMMAND cp ${ASCEND_AUTOGEN_PATH}/op_proto.h ${OPP_INSTALL_DIR}/op_proto/inc
)
