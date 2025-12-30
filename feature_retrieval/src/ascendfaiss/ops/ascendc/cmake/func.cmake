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

function(get_system_info SYSTEM_INFO)
  if (UNIX)
    execute_process(COMMAND grep -i ^id= /etc/os-release OUTPUT_VARIABLE TEMP)
    string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
    set(${SYSTEM_INFO} ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR} PARENT_SCOPE)
  elseif (WIN32)
    message(STATUS "System is Windows. Only for pre-build.")
  else ()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
  endif ()
endfunction()

function(opbuild)
  message(STATUS "Opbuild generating sources")
  cmake_parse_arguments(OPBUILD "" "OUT_DIR;PROJECT_NAME;ACCESS_PREFIX" "OPS_SRC" ${ARGN})
  execute_process(COMMAND ${CMAKE_COMPILE} -fPIC -shared -std=c++11 ${OPBUILD_OPS_SRC} -D_GLIBCXX_USE_CXX11_ABI=0
                  -I ${ASCEND_TOOLKIT_PATH}/include -L ${ASCEND_TOOLKIT_PATH}/lib64 -lexe_graph -lregister -ltiling_api
                  -o ${OPBUILD_OUT_DIR}/libascend_all_ops.so
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("build ops lib info: ${EXEC_INFO}")
    message("build ops lib error: ${EXEC_ERROR}")
    message(FATAL_ERROR "opbuild run failed!")
  endif()
  set(proj_env "")
  set(prefix_env "")
  if (NOT "${OPBUILD_PROJECT_NAME}x" STREQUAL "x")
    set(proj_env "OPS_PROJECT_NAME=${OPBUILD_PROJECT_NAME}")
  endif()
  if (NOT "${OPBUILD_ACCESS_PREFIX}x" STREQUAL "x")
    set(prefix_env "OPS_DIRECT_ACCESS_PREFIX=${OPBUILD_ACCESS_PREFIX}")
  endif()
  execute_process(COMMAND ${proj_env} ${prefix_env} ${ASCEND_TOOLKIT_PATH}/toolkit/tools/opbuild/op_build
                          ${OPBUILD_OUT_DIR}/libascend_all_ops.so ${OPBUILD_OUT_DIR}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("opbuild ops info: ${EXEC_INFO}")
    message("opbuild ops error: ${EXEC_ERROR}")
  endif()
  message(STATUS "Opbuild generating sources - done")
endfunction()

function(add_ops_info_target)
  cmake_parse_arguments(OPINFO "" "TARGET;OPS_INFO;OUTPUT;INSTALL_DIR" "" ${ARGN})
  get_filename_component(opinfo_file_path "${OPINFO_OUTPUT}" DIRECTORY)
  add_custom_command(OUTPUT ${OPINFO_OUTPUT}
      COMMAND mkdir -p ${opinfo_file_path}
      COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${INI_2_JSON_PY}
              ${OPINFO_OPS_INFO} ${OPINFO_OUTPUT}
  )
  add_custom_target(${OPINFO_TARGET} ALL
      DEPENDS ${OPINFO_OUTPUT}
  )
  install(FILES ${OPINFO_OUTPUT}
          DESTINATION ${OPINFO_INSTALL_DIR}
  )
endfunction()

function(add_ops_compile_options OP_TYPE)
  cmake_parse_arguments(OP_COMPILE "" "OP_TYPE" "COMPUTE_UNIT;OPTIONS" ${ARGN})
  file(APPEND ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
       "${OP_TYPE},${OP_COMPILE_COMPUTE_UNIT},${OP_COMPILE_OPTIONS}\n")
endfunction()

function(add_ops_impl_target)
  cmake_parse_arguments(OPIMPL "" "TARGET;OPS_INFO;IMPL_DIR;OUT_DIR;INSTALL_DIR" "OPS_BATCH;OPS_ITERATE" ${ARGN})
  add_custom_command(OUTPUT ${OPIMPL_OUT_DIR}/.impl_timestamp_${OPIMPL_TARGET}
      COMMAND mkdir -m 700 -p ${OPIMPL_OUT_DIR}/dynamic
      COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${ASENDC_IMPL_BUILD_PY}
              ${OPIMPL_OPS_INFO}
              \"${OPIMPL_OPS_BATCH}\" \"${OPIMPL_OPS_ITERATE}\"
              ${OPIMPL_IMPL_DIR}
              ${OPIMPL_OUT_DIR}/dynamic
              ${ASCEND_AUTOGEN_PATH}

      COMMAND rm -rf ${OPIMPL_OUT_DIR}/.impl_timestamp_${OPIMPL_TARGET}
      COMMAND touch ${OPIMPL_OUT_DIR}/.impl_timestamp_${OPIMPL_TARGET}
      DEPENDS ${OPIMPL_OPS_INFO}
              ${ASENDC_IMPL_BUILD_PY}
  )
  add_custom_target(${OPIMPL_TARGET} ALL
      DEPENDS ${OPIMPL_OUT_DIR}/.impl_timestamp_${OPIMPL_TARGET})
endfunction()
