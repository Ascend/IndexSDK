cmake_minimum_required(VERSION 3.14.1)

project(TopoDetect CXX)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/tool_chain.cmake)

set(TOOL_LIB_SRC
${CMAKE_CURRENT_LIST_DIR}/TopoDetectParam.cpp
${CMAKE_CURRENT_LIST_DIR}/TopoDetectParamVerify.cpp
${CMAKE_CURRENT_LIST_DIR}/TopoDetectResult.cpp
${CMAKE_CURRENT_LIST_DIR}/TopoTestPackage.cpp
${CMAKE_CURRENT_LIST_DIR}/TopoDetectChecker.cpp
${CMAKE_CURRENT_LIST_DIR}/DeviceCpuSet.cpp
${CMAKE_CURRENT_LIST_DIR}/DetectModel.cpp
${CMAKE_CURRENT_LIST_DIR}/OckMemoryCopyTestFunOp.cpp
)

set(TOOL_SRC
${CMAKE_CURRENT_LIST_DIR}/TopoDetect.cpp
)
set(TARGET_INSTALL_TOOLS ${OUTPUT_PATH}/memory_bridge/tools)

add_library(TopoDetectImpl SHARED ${TOOL_LIB_SRC})
target_link_libraries(TopoDetectImpl PUBLIC ascendcl ock_hmm)

include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/tool_chain_exe.cmake)
add_executable(TopoDetect ${TOOL_SRC} ${TOOL_LIB_SRC})
target_link_libraries(TopoDetect PUBLIC ascendcl ock_hmm)

file(MAKE_DIRECTORY ${TARGET_INSTALL_TOOLS})
file(MAKE_DIRECTORY ${OUTPUT_PATH}/memory_bridge/lib)
set(EXECUTABLE_OUTPUT_PATH "${OUTPUT_PATH}/memory_bridge/tools")
install(TARGETS TopoDetect DESTINATION ${TARGET_INSTALL_TOOLS}/ PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
install(TARGETS TopoDetectImpl DESTINATION ${OUTPUT_PATH}/memory_bridge/lib/ PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)