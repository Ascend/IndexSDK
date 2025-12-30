cmake_minimum_required(VERSION 3.15)
project(disksearch)

set(CMAKE_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
set(PROCESS_TYPE ${CMAKE_HOST_SYSTEM_PROCESSOR})

# 工程本身include
include_directories(${PROJECT_SOURCE_DIR}/include)

#OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No OpenMP support")
endif()

# 使用openBlas库
include_directories("/opt/OpenBLAS/include/") # openBlas库

#Main compiler/linker settings
set(ARCH_CFLAGS "")
if (${PROCESS_TYPE} STREQUAL "aarch64")
    set(ARCH_CFLAGS "-march=armv8.2-a+fp+simd+crc")
endif ()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_CFLAGS} -fPIC -ftree-vectorize -fopenmp -fopenmp-simd -funroll-loops -flax-vector-conversions")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -Ofast")

# 新增严格编译选项
add_compile_options(
    -fstack-protector-all
    -fno-common
    -D_FORTIFY_SOURCE=2
    -fno-strict-aliasing
    -Wall
    -Wreturn-type
    -Wfloat-equal
    -Wextra
    -fbuiltin
    -lp64
    -fvisibility=hidden # this compile option hides all symbols within the library by default
)

add_link_options(
    -Wl,-z,relro # for relocation read only
    -Wl,-z,now # immediate binding to resolve all symbols at startup (combine with relro for "BIND NOW" security check)
)

if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_link_options(-s) # strip all symbols when releasing, but not in debug
endif()

set(CMAKE_SKIP_BUILD_RPATH TRUE) # remove rpath info from library (Runpath security check)

add_subdirectory(src)