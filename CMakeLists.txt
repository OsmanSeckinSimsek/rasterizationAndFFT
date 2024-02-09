cmake_minimum_required(VERSION 3.22)

project(rasterizationAndFFT CXX)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

list(APPEND CMAKE_MODULE_PATH
    "${PROJECT_SOURCE_DIR}/cmake"
    "${PROJECT_SOURCE_DIR}/main/cmake"
)

option(BUILD_TESTING "build unit and integration tests" ON)
option(BUILD_ANALYTICAL "build analytical solution" ON)

option(GPU_DIRECT "Enable CUDA-aware MPI communication" OFF)
set(HEFFTE_PATH "$ENV{HOME}/lib_installed/heffte" CACHE PATH "Path to HeFFTe installation")
set(FFTW_PATH "$ENV{HOME}/lib_installed/fftw" CACHE PATH "Path to FFTW installation")

set(CSTONE_DIR ${PROJECT_SOURCE_DIR}/cstone/include)
set(CSTONE_TEST_DIR ${PROJECT_SOURCE_DIR}/cstone/test)

find_package(MPI)
find_package(OpenMP)

# if INTERFACE_LINK_OPTIONS for OpenMP_CXX is not set, we set
# INTERFACE_LINK_OPTIONS = INTERFACE_COMPILE_OPTIONS
# i.e. we add e.g. the -fopenmp flag to the linker
get_property(OpenMP_CXX_INTERFACE_LINK_OPTIONS TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_OPTIONS)

if(NOT OpenMP_CXX_INTERFACE_LINK_OPTIONS)
    # fill in using INTERFACE_COMPILE_OPTIONS
    get_property(OpenMP_CXX_INTERFACE_LINK_OPTIONS TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_OPTIONS ${OpenMP_CXX_INTERFACE_LINK_OPTIONS})
endif()

include(CTest)
include(CheckLanguage)

set(default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    set(CMAKE_CUDA_STANDARD 17)
endif()

check_language(HIP)

if(CMAKE_HIP_COMPILER AND NOT CMAKE_CUDA_COMPILER)
    enable_language(HIP)
    find_package(hip)
    set(CMAKE_HIP_STANDARD 17)
endif()

option(ENABLE_H5PART "Enable HDF5 IO using the H5Part library" ON)

if(ENABLE_H5PART)
    set(HDF5_PREFER_PARALLEL true)
    find_package(HDF5)

    if(HDF5_FOUND)
        add_subdirectory(./extern/h5part)
    else()
        message(STATUS "No HDF5 support")
        set(ENABLE_H5PART OFF)
    endif()
endif()

add_subdirectory(cstone)
add_subdirectory(extern/io)
add_subdirectory(main)