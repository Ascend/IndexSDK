CMAKE_MINIMUM_REQUIRED(VERSION 3.13)

PROJECT(ops)

OPTION(BUILD_ASCENDC  "Build ascendc"  ON)

IF (NOT DEFINED NPU_TYPE)
    MESSAGE(FATAL_ERROR  "Please define NPU_TYPE via cmake command line option: -DNPU_TYPE=")
ENDIF()
IF (NOT DEFINED ASCEND_TOOLKIT_PATH)
    SET(ASCEND_TOOLKIT_PATH  /usr/local/Ascend/ascend-toolkit/latest)
ENDIF()
IF (NOT DEFINED DRIVER_HOME)
    SET(DRIVER_HOME  /usr/local/Ascend)
ENDIF()

INCLUDE(${CMAKE_CURRENT_LIST_DIR}/cmake/config.cmake)

ADD_SUBDIRECTORY(cpukernel)
ADD_SUBDIRECTORY(op_proto)
ADD_SUBDIRECTORY(tbe)

IF(BUILD_ASCENDC)
    MESSAGE(STATUS "OPS build with ascendc")
    ADD_SUBDIRECTORY(ascendc)
    SET(ALL_MODULES ${OP_PROTO_TARGET} ai_core_ops_config_json ${AICPU_KERNEL_TARGET} aicpu_ops_config_json optiling_compat cust_optiling cust_op_proto_ascendc)
    ADD_CUSTOM_TARGET(${RUN_TARGET} ALL DEPENDS ${ALL_MODULES})
    ADD_CUSTOM_COMMAND(TARGET ${RUN_TARGET}
        PRE_BUILD
        COMMAND mkdir -p ${OPP_INSTALL_DIR}/op_impl/cpu/aicpu_kernel/impl/
        COMMAND mkdir -p ${OPP_INSTALL_DIR}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl
        COMMAND mkdir -p ${OPP_INSTALL_DIR}/op_impl/vector_core/
        COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/tbe/impl/*.py ${OPP_INSTALL_DIR}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/scripts/install.sh ./makepkg/
        COMMAND python3 ${MERGE_JSON_PY} ./ascendc/op_kernel/tbe/op_info_cfg/ai_core/ ${OPP_INSTALL_DIR}/op_impl/ai_core/tbe/config/
        COMMAND cp -r ./ascendc/op_kernel/tbe/dynamic ${OPP_INSTALL_DIR}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/ascendc/op_kernel/*.cpp ${OPP_INSTALL_DIR}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl/dynamic
        COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/ascendc/op_kernel/*.h ${OPP_INSTALL_DIR}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl/dynamic
        COMMAND mkdir -p ${OPP_INSTALL_DIR}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_HOST_SYSTEM_PROCESSOR}/
        COMMAND cp ./ascendc/op_host/libcust_opmaster_rt2.0.so ${OPP_INSTALL_DIR}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_HOST_SYSTEM_PROCESSOR}/
        COMMAND cp -d ./ascendc/op_host/liboptiling.so ${OPP_INSTALL_DIR}/op_impl/ai_core/tbe/op_tiling/
        COMMAND mkdir -p ${OPP_INSTALL_DIR}/op_proto/lib/linux/${CMAKE_HOST_SYSTEM_PROCESSOR}/
        COMMAND cp ./ascendc/op_host/libcust_opsproto_rt2.0.so ${OPP_INSTALL_DIR}/op_proto/lib/linux/${CMAKE_HOST_SYSTEM_PROCESSOR}/
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/makeself/makeself.sh --gzip --complevel 4 --nomd5 --sha256 --header ${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/makeself/makeself-header.sh ./makepkg ${RUN_TARGET} "version:1.0" ./install.sh)
ELSE()
    SET(ALL_MODULES ${OP_PROTO_TARGET} ai_core_ops_config_json ${AICPU_KERNEL_TARGET} aicpu_ops_config_json)
    ADD_CUSTOM_TARGET(${RUN_TARGET} ALL DEPENDS ${ALL_MODULES})
    ADD_CUSTOM_COMMAND(TARGET ${RUN_TARGET}
        PRE_BUILD
        COMMAND mkdir -p ${OPP_INSTALL_DIR}/op_impl/cpu/aicpu_kernel/impl/
        COMMAND mkdir -p ${OPP_INSTALL_DIR}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl
        COMMAND mkdir -p ${OPP_INSTALL_DIR}/op_impl/vector_core/
        COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/tbe/impl/*.py ${OPP_INSTALL_DIR}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/scripts/install.sh ./makepkg/
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/makeself/makeself.sh --gzip --complevel 4 --nomd5 --sha256 --header ${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/makeself/makeself-header.sh ./makepkg ${RUN_TARGET} "version:1.0" ./install.sh)
ENDIF()