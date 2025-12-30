cmake_minimum_required(VERSION 3.14.1)

project(ock_vsa_hpp CXX)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/tool_chain.cmake)

aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/vsa/neighbor/hpp/kernel/impl VSA_SRC)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/vsa/neighbor/hpp/impl VSA_SRC)
aux_source_directory(${CMAKE_CURRENT_LIST_DIR}/vsa/neighbor/hpp VSA_SRC)

add_library(ock_vsa_hpp SHARED ${VSA_SRC})

target_link_libraries(ock_vsa_hpp PUBLIC ock_hmm ock_hcps_pier c_sec ascendcl pthread)
file(MAKE_DIRECTORY ${OUTPUT_PATH}/vsa)
set(TARGET_INSTALL_LIB ${OUTPUT_PATH}/vsa/lib)
set_target_properties(ock_vsa_hpp PROPERTIES VERSION 23.0.0 SOVERSION 1)
install(TARGETS ock_vsa_hpp DESTINATION ${TARGET_INSTALL_LIB}/ PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)