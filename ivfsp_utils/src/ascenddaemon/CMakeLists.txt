INCLUDE(${PROJECT_SRC_ROOT}/cmake/AscendToolchain.cmake)
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_SKIP_BUILD_RPATH  TRUE)

INCLUDE(${PROJECT_SRC_ROOT}/cmake/utils.cmake)

SET(TARGET_DAEMON ascendfaissdaemon)
SET(ASCEND_DEVICE_DIR ${PROJECT_SRC_ROOT}/ascenddaemon)

FILE(GLOB_RECURSE ASCEND_DEVICE_SRC_FILE
    ${ASCEND_DEVICE_DIR}/*.cpp
    ${PROJECT_SRC_ROOT}/ascend/utils/fp16.cpp
    ${PROJECT_SRC_ROOT}/common/*.pb.cc
    ${PROJECT_SRC_ROOT}/common/*.cpp)

INCLUDE_DIRECTORIES(
    ${PROJECT_SRC_ROOT}
    ${PROJECT_SRC_ROOT}/common
    ${PROJECT_SRC_ROOT}/ascend/utils
    ${PROTOBUF_AARCH64_HOME}/include
    ${ASCEND_MINIOS_HOME}/include
    ${DRIVER_HOME}/driver/include/dvpp
    ${DRIVER_HOME}/driver/kernel/inc/driver)

LINK_DIRECTORIES(
    ${DRIVER_HOME}/develop/lib64
    ${ASCEND_MINIOS_HOME}/lib64)

ADD_COMPILE_OPTIONS(-fPIE -fstack-protector-all -Wall -fno-strict-aliasing -Wreturn-type
                    -D_FORTIFY_SOURCE=2 -frename-registers -fpeel-loops -fopenmp -O3)
ADD_LINK_OPTIONS(-Wl,-z,relro,-z,now,-z,noexecstack -s -pie)
ADD_DEFINITIONS(-DUSE_ACL_INTERFACE_V2)

SET(RPATH "/lib/aarch64-linux-gnu:${ASCEND_MINIOS_HOME}/lib64:${DRIVER_HOME}/develop/lib64:${DRIVER_HOME}/driver/lib64")
SET(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath-link,${RPATH}")

ADD_EXECUTABLE(${TARGET_DAEMON}  ${ASCEND_DEVICE_SRC_FILE})
redefine_file_macro(${TARGET_DAEMON}  ${PROJECT_SRC_ROOT})

TARGET_LINK_LIBRARIES(${TARGET_DAEMON} ${PROTOBUF_AARCH64_HOME}/lib/libprotobuf.a c_sec ascendcl ascend_hal)
INSTALL(TARGETS  ${TARGET_DAEMON}  DESTINATION  ${CMAKE_INSTALL_PREFIX})