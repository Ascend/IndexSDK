PROJECT(mxIndexUnitCases)

IF(COVERAGE)
    GET_DIRECTORY_PROPERTY(CUR_OPTS COMPILE_OPTIONS)
    LIST(REMOVE_ITEM CUR_OPTS -fprofile-arcs -ftest-coverage)
    SET_DIRECTORY_PROPERTIES(PROPERTIES COMPILE_OPTIONS "${CUR_OPTS}")
ENDIF()

ADD_CUSTOM_TARGET(mxindex-lcov
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

ADD_CUSTOM_COMMAND(TARGET mxindex-lcov  # 进行lcov 的代码覆盖情况统计
    COMMAND echo "========= LCOV html is creating ... ===="
    COMMAND lcov --rc lcov_branch_coverage=1 -c -d \"${CMAKE_SOURCE_DIR}\" -o all.info >/dev/null
    COMMAND lcov --rc lcov_branch_coverage=1 --remove all.info '${CMAKE_SOURCE_DIR}/ascenddaemon/impl_custom/*' '${CMAKE_SOURCE_DIR}/ascend/custom/impl/*' '${CMAKE_SOURCE_DIR}/ascendhost/src/impl/*' -o all.info
    COMMAND lcov --rc lcov_branch_coverage=1 -e all.info \"${CMAKE_SOURCE_DIR}/*\" -o project.info
    COMMAND lcov --rc lcov_branch_coverage=1 -r project.info \"${CMAKE_SOURCE_DIR}/test/*\" -o coverage.info   # 删除不需要统计的文件
    COMMAND lcov --rc lcov_branch_coverage=1 -r coverage.info '${CMAKE_SOURCE_DIR}/faiss/*' -o coverage.info
    COMMAND genhtml --rc genhtml_branch_coverage=1 coverage.info -o cover_report
    COMMAND echo "========= LCOV html is created successfully. ===="
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

ADD_SUBDIRECTORY(DTascendhost)