CMAKE_MINIMUM_REQUIRED(VERSION 3.20)
PROJECT(ascenddevice_test)

SET(DEVICE_INSTALL_HOME  ${CMAKE_CURRENT_LIST_DIR}/../../build/device                       CACHE STRING   "")
SET(GTEST_DIR            ${CMAKE_CURRENT_LIST_DIR}/../../third_party/googletest/googletest  CACHE STRING   "")
SET(ASCEND_TOOLKIT_PATH  /usr/local/Ascend/ascend-toolkit/latest                            CACHE STRING   "")
SET(DRIVER_HOME          /usr/local/Ascend/driver                                           CACHE STRING   "")

INCLUDE(${CMAKE_CURRENT_LIST_DIR}/../../cmake/AscendToolchain.cmake)

ADD_COMPILE_OPTIONS(-fPIE -fstack-protector-all -Wall -fno-strict-aliasing -Wreturn-type
                    -D_FORTIFY_SOURCE=2 -frename-registers -fpeel-loops -fopenmp -O3)
ADD_LINK_OPTIONS(-Wl,-z,relro,-z,now,-z,noexecstack -s -pie -fopenmp)

SET(TARGET_GTEST  gtest)
FILE(GLOB  GTEST_SRCS  ${GTEST_DIR}/src/*.cc)
ADD_LIBRARY(${TARGET_GTEST}  STATIC  ${GTEST_SRCS})
TARGET_INCLUDE_DIRECTORIES(${TARGET_GTEST}  PRIVATE
                           ${GTEST_DIR}/include  ${GTEST_DIR})
TARGET_LINK_LIBRARIES(${TARGET_GTEST}  pthread)

INCLUDE_DIRECTORIES(
        ${CMAKE_CURRENT_LIST_DIR}/../../
        ${CMAKE_CURRENT_LIST_DIR}/../../common
        ${GTEST_DIR}/include/
        ${DEVICE_INSTALL_HOME}/include
        ${DRIVER_HOME}/driver/include/dvpp
        ${DRIVER_HOME}/driver/kernel/inc/driver)
LINK_DIRECTORIES(
        ${DEVICE_INSTALL_HOME}/lib
        ${DRIVER_HOME}/develop/lib64)
LINK_LIBRARIES(${TARGET_GTEST}  c_sec  ascendcl  ascend_hal)

SET(RPATH "/lib/aarch64-linux-gnu:${${DEVICE_INSTALL_HOME}/lib}:${DRIVER_HOME}/develop/lib64:${DRIVER_HOME}/driver/lib64")
SET(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath-link,${RPATH}")

# Test executables
ADD_EXECUTABLE(TestIndexILFlat  TestIndexILFlat.cpp)
