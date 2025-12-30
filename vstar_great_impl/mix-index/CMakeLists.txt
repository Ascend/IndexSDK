cmake_minimum_required(VERSION 3.16.0)
project(ivfsp)

SET(TARGET_LIBRARY ascendsearch)
# Compiling options
add_compile_options(-std=c++17)
ADD_COMPILE_OPTIONS(-fPIE -fstack-protector-all -fPIC -Wall -g -O3 -w)
FIND_PACKAGE(OpenMP REQUIRED)
IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS          "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSE()
    MESSAGE(WARNING "OPENMP NOT FOUND")
ENDIF()
SET(CMAKE_SKIP_BUILD_RPATH  TRUE)

add_compile_definitions(USE_ACL_INTERFACE_V2=1)
add_compile_definitions(HOSTCPU)

SET(PROCESS_TYPE ${CMAKE_HOST_SYSTEM_PROCESSOR})
MESSAGE(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR=${CMAKE_HOST_SYSTEM_PROCESSOR}")
IF (${PROCESS_TYPE} STREQUAL "x86_64")
    ADD_COMPILE_OPTIONS(-mavx2 -msse4.1)
ENDIF ()

IF (${PROCESS_TYPE} STREQUAL "aarch64")
    ADD_COMPILE_OPTIONS(-march=armv8.2-a+dotprod)
ENDIF ()

SET(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR})
SET(ASCEND_TOOLKIT_PATH /usr/local/Ascend/ascend-toolkit/latest)
SET(DRIVER_HOME  /usr/local/Ascend/driver)


FILE(GLOB_RECURSE SP_SRC_FILES ${PROJECT_SRC_ROOT}/src/*.cpp)

MESSAGE(STATUS "ACL_ONLINE_MODE=${ACL_ONLINE_MODE}")

if (ACL_ONLINE_MODE)
    message(STATUS "Build aclnn calling mode.")
    add_compile_definitions(USE_ACL_NN_INTERFACE=1)
    include_directories(
            ${PROJECT_SRC_ROOT}/include
            ${ASCEND_TOOLKIT_PATH}/runtime/include
            ${ASCEND_TOOLKIT_PATH}/atc/include
            ${ASCEND_TOOLKIT_PATH}/opp/vendors/customize/op_api/include
    )
    LINK_DIRECTORIES(
            /usr/local/Ascend/ascend-toolkit/latest/${PROCESS_TYPE}-linux/lib64
            ${ASCEND_TOOLKIT_PATH}/opp/vendors/customize/op_api/lib
    )
    ADD_LIBRARY(${TARGET_LIBRARY} SHARED  ${SP_SRC_FILES})
    TARGET_LINK_LIBRARIES(${TARGET_LIBRARY}
            ascendcl acl_op_compiler nnopbase cust_opapi stdc++)

    INSTALL(TARGETS ${TARGET_LIBRARY} DESTINATION ${PROJECT_SRC_ROOT}/lib)

else ()
    message(STATUS "Build offline calling mode.")
    include_directories(
            ${PROJECT_SRC_ROOT}/include
            ${ASCEND_TOOLKIT_PATH}/${PROCESS_TYPE}-linux/include
            ${DRIVER_HOME}/include/dvpp
            ${ASCEND_TOOLKIT_PATH}/acllib/include
	        ${PROJECT_SRC_ROOT}/third_party/include/
	        ${PROJECT_SRC_ROOT}/kgn/  
	        ${PROJECT_SRC_ROOT}/kgn/libc/            
    )
    LINK_DIRECTORIES(
            ${ASCEND_TOOLKIT_PATH}/${PROCESS_TYPE}-linux/lib64
            ${ASCEND_TOOLKIT_PATH}/acllib/lib64
	        ${PROJECT_SRC_ROOT}/third_party/lib/
	        ${PROJECT_SRC_ROOT}/kgn/libc/build/
    )


    ADD_LIBRARY(${TARGET_LIBRARY} SHARED  ${SP_SRC_FILES})

    TARGET_LINK_LIBRARIES(${TARGET_LIBRARY} acl_retr ascendcl)# graphsearch kgn)

    INSTALL(TARGETS ${TARGET_LIBRARY} DESTINATION ${PROJECT_SRC_ROOT}/lib)
endif ()

message(PROJECT_SOURCE_ROOT="${PROJECT_SOURCE_ROOT}")
message(PROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")