CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
PROJECT(ascendhost)

#INCLUDE(${CMAKE_CURRENT_LIST_DIR}/../cmake/utils.cmake)

IF (NOT DEFINED ASCEND_HOME)
    SET(ASCEND_HOME  /usr/local/Ascend  CACHE  STRING "")
ENDIF()
IF (NOT DEFINED FAISS_HOME)
    SET(FAISS_HOME  /usr/local/  CACHE  STRING "")
ENDIF()
IF (NOT DEFINED ASCEND_TOOLKIT_PATH)
    SET(ASCEND_TOOLKIT_PATH  /usr/local/Ascend/ascend-toolkit/latest)
ENDIF()

SET(CMAKE_CXX_STANDARD 11)
#SET(CMAKE_C_COMPILER   gcc)
#SET(CMAKE_CXX_COMPILER g++)
SET(CMAKE_SKIP_BUILD_RPATH  TRUE)

ADD_COMPILE_OPTIONS(
    -fPIC  -fstack-protector-all  -Wall  -fno-strict-aliasing  -Wreturn-type
    -D_FORTIFY_SOURCE=2  -frename-registers  -fpeel-loops  -fopenmp  -O3
    -DHOSTCPU  -DUSE_ACL_INTERFACE_V2)

ADD_LINK_OPTIONS(
    -Wl,-z,relro  -Wl,-z,now  -Wl,-z,noexecstack  -s)

SET(TARGET_LIBRARY         ascendfaiss_host)
SET(TARGET_LIBRARY_RENAME  ascendsearch)
SET(RETRIEVAL_COMMON_DIR   ../)
SET(TARGET_DAEMON_STATIC ascendsearch_static)
IF(HITEST)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_COMPILE hitestwrapper)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_LINK hitestwrapper)
ENDIF()
FILE(GLOB_RECURSE HOST_COMMON_SRC_FILE
    ${RETRIEVAL_COMMON_DIR}/ascend/*.cpp
    ${RETRIEVAL_COMMON_DIR}/ascend/utils/*.cpp
    ${RETRIEVAL_COMMON_DIR}/ascend/impl/*.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/*.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/impl/*.cpp
    ${RETRIEVAL_COMMON_DIR}/common/threadpool/*.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/*.cpp
    ${RETRIEVAL_COMMON_DIR}/ascendhost/src/rpc-local/*.cpp)


# CMAKE_SOURCE_DIR: top level of the source tree
SET(ASCENDFAISS_HOST_SRC
    ${RETRIEVAL_COMMON_DIR}/ascend/AscendIndex.cpp
    ${RETRIEVAL_COMMON_DIR}/ascend/impl/AscendIndexImpl.cpp
    ${RETRIEVAL_COMMON_DIR}/ascend/utils/fp16.cpp
    ${RETRIEVAL_COMMON_DIR}/ascend/utils/MergeResUtils.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/StandardAscendResources.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/impl/AuxIndexStructures.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/impl/Index.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/impl/IndexIVF.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/AscendMemory.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/AscendOpDesc.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/AscendOperator.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/AscendStackMemory.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/AscendUtils.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/MemorySpace.cpp
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/TopkOp.cpp
    ${RETRIEVAL_COMMON_DIR}/ascendhost/src/rpc-local/AscendRpcLocal.cpp
    ${RETRIEVAL_COMMON_DIR}/common/threadpool/AscendThreadPool.cpp
)

INCLUDE_DIRECTORIES(ASCENDFAISS_HOST_INC_DIR
    ${RETRIEVAL_COMMON_DIR}/
    ${RETRIEVAL_COMMON_DIR}/ascend
    ${RETRIEVAL_COMMON_DIR}/ascend/utils
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/impl
    ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils
    ${RETRIEVAL_COMMON_DIR}/common
    ${RETRIEVAL_COMMON_DIR}/ascendhost/include
    ${RETRIEVAL_COMMON_DIR}/ascendhost/include/impl
    ${RETRIEVAL_COMMON_DIR}/ascendhost/include/index
    ${FAISS_HOME}/include
    ${ASCEND_TOOLKIT_PATH}/include
    ${ASCEND_HOME}/driver/include/dvpp
    "/usr/local/include/"
)

LINK_DIRECTORIES(
    ${FAISS_HOME}/lib
    ${ASCEND_TOOLKIT_PATH}/lib64
)
ADD_LIBRARY(objdaemon OBJECT ${HOST_COMMON_SRC_FILE})
ADD_LIBRARY(${TARGET_DAEMON_STATIC} STATIC $<TARGET_OBJECTS:objdaemon>)
#ADD_LIBRARY(${TARGET_LIBRARY}  SHARED  ${ASCENDFAISS_HOST_SRC})
#ADD_EXECUTABLE(${TARGET_LIBRARY} $<TARGET_OBJECTS:objdaemon>)
#SET_TARGET_PROPERTIES(${TARGET_DAEMON_STATIC} PROPERTIES OUTPUT_NAME ${TARGET_LIBRARY})


#GET_FILENAME_COMPONENT(ASCENDFAISS_DIR   ${CMAKE_CURRENT_LIST_DIR}/..  ABSOLUTE)
#redefine_file_macro(${TARGET_LIBRARY}  ${ASCENDFAISS_DIR})
#TARGET_LINK_LIBRARIES(${TARGET_LIBRARY}  faiss  ascendcl  c_sec  pthread)
#SET_TARGET_PROPERTIES(${TARGET_LIBRARY}  PROPERTIES  OUTPUT_NAME  ${TARGET_LIBRARY_RENAME})
#INSTALL(TARGETS  ${TARGET_LIBRARY}  DESTINATION  host/lib)
#INSTALL(TARGETS  ${TARGET_DAEMON_STATIC}  DESTINATION  host/lib)
