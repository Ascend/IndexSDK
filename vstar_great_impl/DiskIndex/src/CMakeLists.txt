set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# OpenGauss适配版本出包的源码
set(OPGS_LIB ${PROJECT_NAME}_opgs)

set(OPGS_VER_CPP_SOURCES MathUtils.cpp FixedChunkPQTable.cpp Adapter/OpenGaussAdapter.cpp)

add_library(${OPGS_LIB} SHARED ${OPGS_VER_CPP_SOURCES})

target_link_libraries(
    ${OPGS_LIB}
    PRIVATE /opt/OpenBLAS/lib/libopenblas.so
)

install(TARGETS ${OPGS_LIB} LIBRARY)