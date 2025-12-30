SET(TARGET_LIBRARY ascendfaiss)

INCLUDE(${PROJECT_SRC_ROOT}/cmake/utils.cmake)

FILE(GLOB ASCEND_SRC_FILE
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/impl/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/custom/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/custom/impl/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rpc/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/utils/*.cpp
    ${PROJECT_SRC_ROOT}/common/*.pb.cc
    ${PROJECT_SRC_ROOT}/common/*.cpp
    ${PROJECT_SRC_ROOT}/common/threadpool/*.cpp
)

INCLUDE_DIRECTORIES(
    ${PROJECT_SRC_ROOT}
    ${PROJECT_SRC_ROOT}/common
    ${PROTOBUF_HOME}/include
    ${FAISS_HOME}/include
    ${ASCEND_TOOLKIT_PATH}/include
    ${DRIVER_HOME}/driver/include/dvpp
    ${DRIVER_HOME}/driver/kernel/inc/driver)

LINK_DIRECTORIES(
    ${PROTOBUF_HOME}/lib
    ${ASCEND_TOOLKIT_PATH}/lib64
    ${FAISS_HOME}/lib
    ${DRIVER_HOME}/driver/lib64     # For arm; x86_64 uses ${DRIVER_HOME}/driver/lib64/driver, CANN should solve this prob.
    ${DRIVER_HOME}/driver/lib64/driver)

IF(HITEST)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_COMPILE hitestwrapper)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_LINK hitestwrapper)
ENDIF()

ADD_LIBRARY(${TARGET_LIBRARY} SHARED  ${ASCEND_SRC_FILE})
redefine_file_macro(${TARGET_LIBRARY}  ${PROJECT_SRC_ROOT})

TARGET_LINK_LIBRARIES(${TARGET_LIBRARY}
    openblas faiss ascend_hal acl_retr protobuf c_sec mmpa z dl rt
    -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack -s)

INSTALL(TARGETS  ${TARGET_LIBRARY}  DESTINATION  lib)
