CMAKE_MINIMUM_REQUIRED(VERSION 3.13)

PROJECT(ops)

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

SET(ALL_MODULES ${OP_PROTO_TARGET} ai_core_ops_config_json ${AICPU_KERNEL_TARGET} aicpu_ops_config_json)

ADD_CUSTOM_TARGET(${RUN_TARGET} ALL DEPENDS ${ALL_MODULES})

ADD_CUSTOM_COMMAND(TARGET ${RUN_TARGET}
        PRE_BUILD
        COMMAND mkdir -p ./makepkg/packages/op_impl/custom/cpu/aicpu_kernel/custom_impl
        COMMAND mkdir -p ./makepkg/packages/op_impl/custom/ai_core/tbe/custom_impl
        COMMAND mkdir -p ./makepkg/packages/op_impl/custom/vector_core/tbe/custom_impl
        COMMAND cp -r ${CMAKE_CURRENT_SOURCE_DIR}/tbe/impl/*.py ./makepkg/packages/op_impl/custom/ai_core/tbe/custom_impl
        COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/scripts/install.sh ./makepkg/
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/cmake/util/makeself/makeself.sh --gzip --complevel 4 --nomd5 --sha256 ./makepkg ${RUN_TARGET} "version:1.0" ./install.sh)
