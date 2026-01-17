INCLUDE(${PROJECT_SRC_ROOT}/cmake/AscendToolchain.cmake)
SET(CMAKE_CXX_STANDARD 17)

INCLUDE(${PROJECT_SRC_ROOT}/cmake/utils.cmake)

SET(TARGET_DEVICE ascendfaiss_minios)
FILE(GLOB TARGET_DEVICE_SRC
    ${PROJECT_SRC_ROOT}/common/threadpool/*.cpp
    ${PROJECT_SRC_ROOT}/ascenddaemon/impl_device/IndexILFlat.cpp
    ${PROJECT_SRC_ROOT}/ascenddaemon/ThreadSafeStandardAscendResources.cpp
    ${PROJECT_SRC_ROOT}/ascenddaemon/StandardAscendResources.cpp
    ${PROJECT_SRC_ROOT}/ascenddaemon/utils/DistComputeOpsManager.cpp
    ${PROJECT_SRC_ROOT}/ascenddaemon/utils/ThreadSafeDistComputeOpsManager.cpp
    ${PROJECT_SRC_ROOT}/ascenddaemon/utils/MemorySpace.cpp
    ${PROJECT_SRC_ROOT}/ascenddaemon/utils/TopkOp.cpp
    ${PROJECT_SRC_ROOT}/ascenddaemon/utils/AscendMemory.cpp
    ${PROJECT_SRC_ROOT}/ascenddaemon/utils/AscendUtils.cpp
    ${PROJECT_SRC_ROOT}/ascenddaemon/utils/AscendRWLock.cpp
    ${PROJECT_SRC_ROOT}/ascenddaemon/utils/AscendStackMemory.cpp
    ${PROJECT_SRC_ROOT}/ascenddaemon/utils/AscendOpDesc.cpp
    ${PROJECT_SRC_ROOT}/ascenddaemon/utils/AscendOperator.cpp)

INCLUDE_DIRECTORIES(
    ${PROJECT_SRC_ROOT}
    ${PROJECT_SRC_ROOT}/common
    ${ASCEND_MINIOS_HOME}/include
    ${DRIVER_HOME}/driver/include/dvpp
    ${DRIVER_HOME}/driver/kernel/inc/driver)

LINK_DIRECTORIES(
    ${DRIVER_HOME}/develop/lib64
    ${ASCEND_MINIOS_HOME}/runtime/lib64/stub
    ${ASCEND_MINIOS_HOME}/lib64)

ADD_COMPILE_OPTIONS(-fPIE -fstack-protector-all -Wall -fno-strict-aliasing -Wreturn-type
    -D_FORTIFY_SOURCE=2 -frename-registers -fpeel-loops -fopenmp -O3 -Wfloat-equal -fno-common -Wextra)
ADD_LINK_OPTIONS(-Wl,-z,relro,-z,now,-z,noexecstack -s -pie)
ADD_DEFINITIONS(-DUSE_ACL_INTERFACE_V2)

SET(RPATH "/lib/aarch64-linux-gnu:${ASCEND_MINIOS_HOME}/lib64:${DRIVER_HOME}/develop/lib64:${DRIVER_HOME}/driver/lib64")
SET(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath-link,${RPATH}")

ADD_LIBRARY(${TARGET_DEVICE} SHARED ${TARGET_DEVICE_SRC})
redefine_file_macro(${TARGET_DEVICE}  ${PROJECT_SRC_ROOT})
TARGET_LINK_LIBRARIES(${TARGET_DEVICE} c_sec ascendcl ascend_hal)
INSTALL(TARGETS  ${TARGET_DEVICE}  DESTINATION  device/lib)