cmake_minimum_required(VERSION 3.16.0)
project(indexgreat)

INCLUDE(./cmake/utils.cmake)

# define global variables
set(PROCESS_TYPE ${CMAKE_HOST_SYSTEM_PROCESSOR})
message(STATUS "CMAKE_HOST_SYSTEM_PROCESSOR=${CMAKE_HOST_SYSTEM_PROCESSOR}")

set(TARGET_LIBRARY indexgreat)

set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR})
message(STATUS "PROJECT_SRC_ROOT=${PROJECT_SRC_ROOT}")

set(ASCEND_TOOLKIT_PATH /usr/local/Ascend/ascend-toolkit/latest)
set(SECURE_API ${ASCEND_TOOLKIT_PATH}/${PROCESS_TYPE}-linux/)
set(DRIVER_HOME  /usr/local/Ascend/driver)
set(FAISS_HOME /usr/local/faiss/faiss1.10.0)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SRC_ROOT}/lib)
set(CMAKE_SKIP_BUILD_RPATH TRUE)

# -----------------------------------------------------------------------------------------------------------

# Set compiler info

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

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

add_compile_definitions(USE_ACL_INTERFACE_V2=1)
add_compile_definitions(HOSTCPU)
add_compile_definitions(INFO)
add_compile_options(-std=c++17 -fPIE -fstack-protector-all -falign-jumps=64 -fPIC -D_FORTIFY_SOURCE=2 -Wall -s -g -O3 -Ofast -Wextra -Wfloat-equal -w)
add_link_options(-Wl,-z,relro  -Wl,-z,now  -Wl,-z,noexecstack -s)
# -----------------------------------------------------------------------------------------------------------

message(STATUS "$CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
message(STATUS "$CMAKE_CXX_LINKER_FLAGS = ${CMAKE_EXE_LINKER_FLAGS}")

# Set file path

set (AMODE_DIR ${PROJECT_SRC_ROOT}/../)
set (KMODE_DIR ${PROJECT_SRC_ROOT}/../third_party/)

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
            ${ASCEND_TOOLKIT_PATH}/include
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
        ${FAISS_HOME}/include
)

link_directories(
        ${FAISS_HOME}/lib64
        ${FAISS_HOME}/lib
)

# -----------------------------------------------------------------------------------------------------------
# start building

add_library(${TARGET_LIBRARY}-static STATIC ${KMODE_SRC_FILES} ${AMODE_SRC_FILES})

add_library(${TARGET_LIBRARY} SHARED ${KMODE_SRC_FILES} ${AMODE_SRC_FILES})

set_target_properties(${TARGET_LIBRARY}-static PROPERTIES OUTPUT_NAME ${TARGET_LIBRARY})

get_filename_component(VSTAR_DIR  ${AMODE_DIR}/src  ABSOLUTE)
get_filename_component(GREAT_DIR_SRC  ${KMODE_DIR}/src  ABSOLUTE)
get_filename_component(GREAT_DIR_IMPL  ${KMODE_DIR}/src/impl  ABSOLUTE)
get_filename_component(GREAT_DIR_UTILS  ${KMODE_DIR}/src/utils  ABSOLUTE)
redefine_file_macro(${TARGET_LIBRARY}  ${VSTAR_DIR})
redefine_file_macro(${TARGET_LIBRARY}  ${GREAT_DIR_SRC})
redefine_file_macro(${TARGET_LIBRARY}  ${GREAT_DIR_IMPL})
redefine_file_macro(${TARGET_LIBRARY}  ${GREAT_DIR_UTILS})

# set target link libraries
if (ACL_ONLINE_MODE)
    target_link_libraries(${TARGET_LIBRARY} ascendcl acl_op_compiler nnopbase cust_opapi stdc++)
endif()

target_link_libraries(${TARGET_LIBRARY} faiss acl_retr ascendcl c_sec pthread)