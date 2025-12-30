PROJECT(LIBASCENDCL_TEST)

SET(TARGET_OUT ascendcl_testmain)
SET(CMAKE_CXX_STANDARD 11)
SET(ASCEND_MOCK_PATH ${CMAKE_CURRENT_LIST_DIR}/..)
SET(GTEST_HOME ${ASCEND_MOCK_PATH}/../../../../gtest_install)

ADD_COMPILE_OPTIONS(
        -Wall
        -O0
        -g
        -fno-access-control
        -Wno-pmf-conversions
)

FILE(GLOB SRC_FILE ${CMAKE_CURRENT_LIST_DIR}/*.cpp)

ADD_EXECUTABLE(${TARGET_OUT} ${SRC_FILE})

target_include_directories(${TARGET_OUT} PRIVATE
        ${GTEST_HOME}/include
        ${ASCEND_MOCK_PATH}/
)

target_link_directories(${TARGET_OUT} PRIVATE
        ${GTEST_HOME}/lib
        ${ASCEND_MOCK_PATH}/lib
)

TARGET_LINK_LIBRARIES(${TARGET_OUT} PRIVATE
        gtest_main
        gtest
        ascendcl
)