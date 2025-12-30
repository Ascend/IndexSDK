# General Settings
CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
PROJECT(feature_retrieval)
SET(CMAKE_CXX_STANDARD 14)

# Compiling options
ADD_COMPILE_OPTIONS(-fPIE -fstack-protector-all -fPIC -Wall -O3)
FIND_PACKAGE(OpenMP REQUIRED)
IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS          "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
ELSE()
    MESSAGE(WARNING "OPENMP NOT FOUND")
ENDIF()
SET(CMAKE_SKIP_BUILD_RPATH  TRUE)

# User-settable configurations
SET(ASCEND_HOME                 /usr/local/Ascend            CACHE STRING   "")
SET(DRIVER_HOME                 /usr/local/Ascend            CACHE STRING   "")
SET(ASCEND_TOOLKIT_VERSION      latest                       CACHE STRING   "")
#SET(ASCEND_MINIOS_HOME          /usr/local/AscendMiniOs      CACHE STRING   "")
#SET(PROTOBUF_HOME               /usr/local/protobuf          CACHE STRING   "")
#SET(PROTOBUF_AARCH64_HOME       /opt/aarch64/protobuf        CACHE STRING   "")
SET(FAISS_HOME                  /usr/local                   CACHE STRING   "")
SET(NPU_TYPE                    310                          CACHE STRING   "310 | 310P")
# Relative paths
SET(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR})
SET(ASCEND_TOOLKIT_PATH ${ASCEND_HOME}/ascend-toolkit/${ASCEND_TOOLKIT_VERSION})
# Check user-settable path existence
FOREACH(X IN LISTS
            ASCEND_HOME
            DRIVER_HOME
            ASCEND_TOOLKIT_PATH
            ASCEND_MINIOS_HOME
            PROTOBUF_HOME
            PROTOBUF_AARCH64_HOME
            FAISS_HOME
        )
    IF (NOT EXISTS ${X})
        MESSAGE(FATAL_ERROR "${X} does not exist")
    ENDIF()
ENDFOREACH()

OPTION(BUILD_ASCENDCTL     "Build control cpu host and device daemon"  OFF)
OPTION(BUILD_ASCENDHOST    "Build ascendhost"                          ON)
OPTION(BUILD_ASCENDDEVICE  "Build ascenddevice"                        OFF)
OPTION(BUILD_OPS           "Build ops"                                 OFF)
OPTION(COVERAGE            "COVERAGE"                                  OFF)
OPTION(HITEST              "HITEST"                                    OFF)

# Protocol generation
EXECUTE_PROCESS(COMMAND ${PROTOBUF_HOME}/bin/protoc
        --proto_path=${PROJECT_SRC_ROOT}/common --cpp_out=${PROJECT_SRC_ROOT}/common
        ${PROJECT_SRC_ROOT}/common/AscendIndex.proto
        WORKING_DIRECTORY ${PROJECT_SRC_ROOT})

# DT
IF(COVERAGE)
    SET(GCC_COVERAGE_COMPILE_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
    SET(GCC_COVERAGE_LINK_FLAGS    "-lgcov")

    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")
ENDIF()

IF(BUILD_ASCENDCTL)
    ADD_SUBDIRECTORY(ascend)
    ADD_SUBDIRECTORY(ascenddaemon)
ENDIF()
IF(BUILD_ASCENDHOST)
    ADD_SUBDIRECTORY(ascendhost)
ENDIF()
IF(BUILD_ASCENDDEVICE)
    ADD_SUBDIRECTORY(ascenddevice)
ENDIF()
IF (BUILD_OPS)
    ADD_SUBDIRECTORY(ops)
ENDIF()

IF(COVERAGE)
    ENABLE_TESTING()
    ADD_SUBDIRECTORY(test)
ENDIF()

# Release Headers
INCLUDE(${CMAKE_CURRENT_LIST_DIR}/cmake/AscendRelease.cmake)





