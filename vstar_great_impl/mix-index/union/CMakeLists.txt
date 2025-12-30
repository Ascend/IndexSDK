cmake_minimum_required(VERSION 3.16.0)
project(ascendsearch)

INCLUDE(../ivfsp_impl/ascendfaiss/cmake/utils.cmake)

# define global variables
set(PROCESS_TYPE ${CMAKE_HOST_SYSTEM_PROCESSOR})
message(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR=${CMAKE_HOST_SYSTEM_PROCESSOR}")

set(TARGET_LIBRARY ascendsearch)

set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR})
message(STATUS "PROJECT_SRC_ROOT=${PROJECT_SRC_ROOT}")

set(ASCEND_TOOLKIT_PATH /usr/local/Ascend/ascend-toolkit/latest)
set(SECURE_API ${ASCEND_TOOLKIT_PATH}/${PROCESS_TYPE}-linux/)
set(DRIVER_HOME  /usr/local/Ascend/driver)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SRC_ROOT}/lib)
set(CMAKE_SKIP_BUILD_RPATH TRUE)

IF (NOT DEFINED ASCEND_HOME)
    SET(ASCEND_HOME  /usr/local/Ascend  CACHE  STRING "")
ENDIF()
IF (NOT DEFINED FAISS_HOME)
    SET(FAISS_HOME  /usr/local/faiss  CACHE  STRING "")
ENDIF()
IF (NOT DEFINED ASCEND_TOOLKIT_PATH)
    SET(ASCEND_TOOLKIT_PATH  /usr/local/Ascend/ascend-toolkit/latest)
ENDIF()

# -----------------------------------------------------------------------------------------------------------

# Set compiler info

# make sure OMP package is used
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS          "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "OPENMP NOT FOUND")
endif()

# check for system processor type (x86 or ARM)
if (${PROCESS_TYPE} STREQUAL "x86_64")
    add_compile_options(-msse4.1)
endif ()
if (${PROCESS_TYPE} STREQUAL "aarch64")
    # add_compile_options(-march=armv8.2-a+dotprod)
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl, ${GCC12_PATH}")
    # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl, -z, now")
    # link_directories("/home/dawid/gcc-13.2.0/lib64/")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl -lz") #-gdwarf-2
else() # x86_64
    set(CMAKE_C_COMPILER gcc)
    set(CMAKE_CXX_COMPILER g++)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif ()

add_compile_definitions(USE_ACL_INTERFACE_V2=1)
add_compile_definitions(HOSTCPU)
add_compile_definitions(INFO)
# add_compile_options(-std=c++17 -fPIE -fstack-protector-all -falign-jumps=64 -fPIC -Wall -g -O3 -Ofast -Wextra -Wfloat-equal -w)
add_compile_options(-std=c++17 -fPIE -fstack-protector-all -falign-jumps=64 -fPIC -D_FORTIFY_SOURCE=2 -s
        -O3 -Ofast -Wfloat-equal -fno-common -Wextra -Wall)
add_compile_options(-fPIC  -fstack-protector-all -fno-strict-aliasing
        -D_FORTIFY_SOURCE=2  -frename-registers  -fpeel-loops  -fopenmp  -O3
        -DHOSTCPU  -DUSE_ACL_INTERFACE_V2)
add_link_options(-Wl,-z,relro  -Wl,-z,now  -Wl,-z,noexecstack -s)
# -----------------------------------------------------------------------------------------------------------

message(STATUS "$CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "$CMAKE_CXX_LINKER_FLAGS = ${CMAKE_EXE_LINKER_FLAGS}")

# Set file path

set (AMODE_DIR ${PROJECT_SRC_ROOT}/../)
set (KMODE_DIR ${PROJECT_SRC_ROOT}/../third_party/)
set (RETRIEVAL_COMMON_DIR ../ivfsp_impl/Retrieval_Common/src)
set (RETRIEVAL_ASCEND_HOST_DIR ../ivfsp_impl/ascendfaiss/ascendhost)


include_directories(
        ${SECURE_API}/include
)

link_directories(
        ${SECURE_API}/lib64
)

# define AMode's source file location, decide building in ACLOnlineMode vs ACLOfflineMode
FILE(GLOB_RECURSE AMODE_SRC_FILES ${AMODE_DIR}/src/*.cpp)

MESSAGE(STATUS "ACL_ONLINE_MODE=${ACL_ONLINE_MODE}")

if (ACL_ONLINE_MODE)
    message(STATUS "Build aclnn calling mode.")
    add_compile_definitions(USE_ACL_NN_INTERFACE=1)
    include_directories(
            ${AMODE_DIR}/include
            ${ASCEND_TOOLKIT_PATH}/runtime/include
            ${ASCEND_TOOLKIT_PATH}/atc/include
            ${ASCEND_TOOLKIT_PATH}/opp/vendors/customize/op_api/include
    )
    link_directories(
            ${ASCEND_TOOLKIT_PATH}/opp/vendors/customize/op_api/lib
    )

else ()
    message(STATUS "Build offline calling mode.")
    include_directories(
            ${AMODE_DIR}/include
            ${DRIVER_HOME}/include/dvpp
            ${ASCEND_TOOLKIT_PATH}/acllib/include
    )
    link_directories(
            ${ASCEND_TOOLKIT_PATH}/acllib/lib64
    )
endif ()


# define KMode's source file location
file(GLOB KMODE_SRC_FILES
        ${KMODE_DIR}/src/*.cpp
        ${KMODE_DIR}/src/impl/*.cpp
        ${KMODE_DIR}/src/utils/*.cpp
)

include_directories(
        ${KMODE_DIR}/include
)

# define IVFSP file
file(GLOB_RECURSE HOST_COMMON_SRC_FILE
        ${RETRIEVAL_COMMON_DIR}/ascend/*.cpp
        ${RETRIEVAL_COMMON_DIR}/ascend/utils/*.cpp
        ${RETRIEVAL_COMMON_DIR}/ascend/impl/*.cpp
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/*.cpp
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/impl/*.cpp
        ${RETRIEVAL_COMMON_DIR}/common/threadpool/*.cpp
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/*.cpp
        ${RETRIEVAL_COMMON_DIR}/ascendhost/src/rpc-local/*.cpp
)

SET(ASCENDFAISS_HOST_SRC
        ${RETRIEVAL_COMMON_DIR}/ascend/AscendIndex.cpp
        ${RETRIEVAL_ASCEND_HOST_DIR}/../ascend/AscendMultiIndexSearch.cpp
        ${RETRIEVAL_ASCEND_HOST_DIR}/../ascend/custom/AscendIndexIVFSPSQ.cpp
        ${RETRIEVAL_COMMON_DIR}/ascend/impl/AscendIndexImpl.cpp
        ${RETRIEVAL_ASCEND_HOST_DIR}/../ascend/custom/impl/AscendIndexIVFSPSQImpl.cpp
        ${RETRIEVAL_COMMON_DIR}/ascend/utils/fp16.cpp
        ${RETRIEVAL_COMMON_DIR}/ascend/utils/MergeResUtils.cpp
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/StandardAscendResources.cpp
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/impl/AuxIndexStructures.cpp
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/impl/Index.cpp
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/impl/IndexIVF.cpp
        ${RETRIEVAL_ASCEND_HOST_DIR}/../ascenddaemon/impl_custom/IndexIVFSPSQ.cpp
        ${RETRIEVAL_ASCEND_HOST_DIR}/../ascenddaemon/impl_custom/IndexIVFSPSQL2.cpp
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/AscendMemory.cpp
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/AscendOpDesc.cpp
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/AscendOperator.cpp
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/AscendStackMemory.cpp
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/AscendUtils.cpp
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/MemorySpace.cpp
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/TopkOp.cpp
        ${RETRIEVAL_ASCEND_HOST_DIR}/../ascendhost/src/index_custom/IndexIVFSPSQL2Aicpu.cpp
        ${RETRIEVAL_COMMON_DIR}/ascendhost/src/rpc-local/AscendRpcLocal.cpp
        ${RETRIEVAL_ASCEND_HOST_DIR}/../ascendhost/src/rpc-local/AscendRpcLocalIndexIVFSPSQ.cpp
        ${RETRIEVAL_COMMON_DIR}/common/threadpool/AscendThreadPool.cpp
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils/IoUtil.cpp
        ${RETRIEVAL_ASCEND_HOST_DIR}/../ascenddaemon/IVFSPCodeBookTrainer.cpp
)

INCLUDE_DIRECTORIES(ASCENDFAISS_HOST_INC_DIR
        ${RETRIEVAL_ASCEND_HOST_DIR}/../
        ${RETRIEVAL_COMMON_DIR}/
        ${RETRIEVAL_ASCEND_HOST_DIR}/../ascend
        ${RETRIEVAL_COMMON_DIR}/ascend
        ${RETRIEVAL_COMMON_DIR}/ascend/utils
        ${RETRIEVAL_ASCEND_HOST_DIR}/../ascenddaemon
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/impl
        ${RETRIEVAL_COMMON_DIR}/ascenddaemon/utils
        ${RETRIEVAL_COMMON_DIR}/common
        ${RETRIEVAL_ASCEND_HOST_DIR}/include
        ${RETRIEVAL_COMMON_DIR}/ascendhost/include
        ${RETRIEVAL_ASCEND_HOST_DIR}/include/impl
        ${RETRIEVAL_COMMON_DIR}/ascendhost/include/impl
        ${RETRIEVAL_ASCEND_HOST_DIR}/include/index
        ${RETRIEVAL_COMMON_DIR}/ascendhost/include/index
        ${FAISS_HOME}/include
        /opt/OpenBLAS/include
        ${ASCEND_TOOLKIT_PATH}/include
        ${ASCEND_HOME}/driver/include/dvpp
        "/usr/local/include/"
)

LINK_DIRECTORIES(
        ${FAISS_HOME}/lib
        ${ASCEND_TOOLKIT_PATH}/lib64
)

# -----------------------------------------------------------------------------------------------------------
# start building
ADD_LIBRARY(objdaemon OBJECT ${HOST_COMMON_SRC_FILE})

add_library(${TARGET_LIBRARY}-static STATIC ${KMODE_SRC_FILES} ${AMODE_SRC_FILES} $<TARGET_OBJECTS:objdaemon>)

add_library(${TARGET_LIBRARY} SHARED ${KMODE_SRC_FILES} ${AMODE_SRC_FILES} ${ASCENDFAISS_HOST_SRC})

get_filename_component(ASCENDFAISS_DIR   ${RETRIEVAL_ASCEND_HOST_DIR}/..  ABSOLUTE)
get_filename_component(VSTAR_DIR  ${AMODE_DIR}/src  ABSOLUTE)
get_filename_component(GREAT_DIR_SRC  ${KMODE_DIR}/src  ABSOLUTE)
get_filename_component(GREAT_DIR_IMPL  ${KMODE_DIR}/src/impl  ABSOLUTE)
get_filename_component(GREAT_DIR_UTILS  ${KMODE_DIR}/src/utils  ABSOLUTE)

set_target_properties(${TARGET_LIBRARY}-static PROPERTIES OUTPUT_NAME ${TARGET_LIBRARY})

# set target link libraries
if (ACL_ONLINE_MODE)
    target_link_libraries(${TARGET_LIBRARY} faiss c_sec ascendcl acl_op_compiler nnopbase cust_opapi stdc++)
endif()

target_link_libraries(${TARGET_LIBRARY} faiss c_sec acl_retr ascendcl pthread)

INCLUDE(../ivfsp_impl/ascendfaiss/cmake/AscendRelease.cmake)