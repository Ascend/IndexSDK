INCLUDE(${CMAKE_CURRENT_LIST_DIR}/../toolchain.cmake)

SET(AICPU_INI_2_JSON_PY  ${CMAKE_CURRENT_LIST_DIR}/../cmake/util/aicpu_parser_ini.py)

MACRO(GEN_OPS_INFO_FILE  op_type  output_dir  output_file_name_prefix)
    SET(INI_PATH  ${CMAKE_CURRENT_SOURCE_DIR}/op_info_cfg)
    EXECUTE_PROCESS(COMMAND  ls -1  ${INI_PATH}  OUTPUT_VARIABLE  SUB_DIRS)
    STRING(REPLACE  "\n" ";"  SUB_DIRS  ${SUB_DIRS})
    FOREACH(SUB_DIR  ${SUB_DIRS})
        IF(IS_DIRECTORY  ${INI_PATH}/${SUB_DIR})
            EXECUTE_PROCESS(COMMAND  find  ${INI_PATH}/${SUB_DIR}  -name  "*.ini"  OUTPUT_VARIABLE  INI_FILES)
            IF(NOT "x${INI_FILES}" STREQUAL "x")
                STRING(REPLACE  "\n" "\t"  INI_FILES  ${INI_FILES})
            ENDIF()
            IF(NOT  "x${INI_FILES}"  STREQUAL  "x")
                SET(output_file_name  ${output_file_name_prefix}_aicpu_kernel.json)
                ADD_CUSTOM_COMMAND(OUTPUT ${output_file_name}
                        COMMAND  mkdir -p  ${output_dir}
                        COMMAND  python3  ${AICPU_INI_2_JSON_PY}  ${INI_FILES}  ${output_dir}/${output_file_name})
                SET(OUTPUT_FILES ${OUTPUT_FILES} ${output_file_name})
            ELSE()
                MESSAGE(STATUS "ini file not exists.")
                EXECUTE_PROCESS(COMMAND mkdir -p ${output_dir})
            ENDIF()
        ENDIF()
    ENDFOREACH()
ENDMACRO(GEN_OPS_INFO_FILE)

SET(OUTPUT_FILES "")
GEN_OPS_INFO_FILE("" ${AICPU_OP_INFO_CFG_OUT_DIR} cust)
ADD_CUSTOM_TARGET(aicpu_ops_config_json  ALL DEPENDS  ${OUTPUT_FILES})

# =========================kernel compile============================
SET(AICPU_LIB_SUB_DIR opp/built-in/op_impl/aicpu/aicpu_kernel/lib)
IF (NOT EXISTS ${ASCEND_TOOLKIT_PATH}/${AICPU_LIB_SUB_DIR}/libcpu_kernels_context.a)
    SET(AICPU_LIB_SUB_DIR opp/op_impl/built-in/aicpu/aicpu_kernel/lib)
    IF (NOT EXISTS ${ASCEND_TOOLKIT_PATH}/${AICPU_LIB_SUB_DIR}/libcpu_kernels_context.a)
        MESSAGE(FATAL_ERROR "libcpu_kernels_context.a not found in opp")
    ENDIF()
ENDIF()

FILE(GLOB_RECURSE KERNELS_SRCS KERNELS_SRCS
        ${CMAKE_CURRENT_LIST_DIR}/impl/*.cpp
)

INCLUDE_DIRECTORIES(${ASCEND_TOOLKIT_PATH}/${AICPU_LIB_SUB_DIR}/../inc)
INCLUDE_DIRECTORIES(${DRIVER_HOME}/include/dvpp)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/impl/utils)

SET(LIBRARY_OUTPUT_PATH ${AICPU_OP_IMPL_OUT_DIR})
ADD_LIBRARY(${AICPU_KERNEL_TARGET} SHARED
        ${KERNELS_SRCS}
)

# Compatible with CI CANN version.
SET(CI_NPU_TYPE  ${NPU_TYPE})
IF(NPU_TYPE  STREQUAL  310P)
    IF(NOT  EXISTS  ${ASCEND_TOOLKIT_PATH}/${AICPU_LIB_SUB_DIR}/Ascend${CI_NPU_TYPE})
        SET(CI_NPU_TYPE  710)
    ENDIF()
ENDIF()

TARGET_LINK_LIBRARIES(${AICPU_KERNEL_TARGET} PRIVATE
    -Wl,--whole-archive
    ${ASCEND_TOOLKIT_PATH}/${AICPU_LIB_SUB_DIR}/Ascend${CI_NPU_TYPE}/libascend_protobuf.a
    -Wl,--no-whole-archive
    -s
    -Wl,-Bsymbolic
    -Wl,--exclude-libs=libascend_protobuf.a)

TARGET_LINK_LIBRARIES(${AICPU_KERNEL_TARGET} PRIVATE
        -Wl,--whole-archive
        ${ASCEND_TOOLKIT_PATH}/${AICPU_LIB_SUB_DIR}/Ascend${CI_NPU_TYPE}/libcpu_kernels_context.a
        -Wl,--no-whole-archive)
