function(enableGpuDirect exename)
    if(GPU_DIRECT)
        target_compile_definitions(${exename} PRIVATE USE_GPU_DIRECT)
    endif()
endfunction()

if(RASTER_WITH_NVSHMEM AND NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "RASTER_WITH_NVSHMEM=ON requires a CUDA compiler.")
endif()

# find_package(Heffte PATHS $(HEFFTE_PATH))
# set(HEFFTE_PATH "$ENV{HOME}/lib_installed/heffte" CACHE PATH "Path to HeFFTe installation")
set(exename power_spectrum)
add_executable(${exename} power_spectrum.cpp)
# Get include directories from the Heffte target if available, otherwise fall back to HEFFTE_PATH
if(TARGET Heffte::Heffte)
    get_target_property(HEFFTE_INC_DIR Heffte::Heffte INTERFACE_INCLUDE_DIRECTORIES)
    if(NOT HEFFTE_INC_DIR)
        # Fallback: construct from HEFFTE_PATH
        set(HEFFTE_INC_DIR "${HEFFTE_PATH}/include")
    endif()
else()
    set(HEFFTE_INC_DIR "${HEFFTE_PATH}/include")
endif()
message(STATUS "HEFFTE_INC_DIR=${HEFFTE_INC_DIR}")
message(STATUS "CSTONE_DIR=${CSTONE_DIR}")
target_include_directories(${exename} PUBLIC ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/extern/io ${HEFFTE_INC_DIR})
target_link_libraries(${exename} PRIVATE io OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES} Heffte::Heffte)
install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(CMAKE_CUDA_COMPILER)
    add_executable(${exename}-cuda power_spectrum.cpp mesh.cu)
    target_include_directories(${exename}-cuda PUBLIC ${CMAKE_BINARY_DIR}/main/src  ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/extern/io ${HEFFTE_INC_DIR})
    if(CUDAToolkit_INCLUDE_DIRS)
        target_include_directories(${exename}-cuda PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
        set(cuda_std_extra_includes)
        foreach(cuda_inc_dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
            if(EXISTS "${cuda_inc_dir}/cccl/cuda/std")
                list(APPEND cuda_std_extra_includes "${cuda_inc_dir}/cccl")
            elseif(EXISTS "${cuda_inc_dir}/cuda/std")
                list(APPEND cuda_std_extra_includes "${cuda_inc_dir}")
            endif()
        endforeach()
        if(cuda_std_extra_includes)
            list(REMOVE_DUPLICATES cuda_std_extra_includes)
            target_include_directories(${exename}-cuda PUBLIC ${cuda_std_extra_includes})
        endif()
    endif()
    target_compile_definitions(${exename}-cuda PRIVATE USE_CUDA)
    target_link_libraries(${exename}-cuda PRIVATE cstone_gpu io OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES} CUDA::cudart Heffte::Heffte)
    if(RASTER_WITH_NVSHMEM)
        target_compile_definitions(${exename}-cuda PRIVATE USE_NVSHMEM)
        # Accept NVSHMEM_HOME as CMake variable or environment variable
        if(NOT NVSHMEM_HOME)
            set(NVSHMEM_HOME $ENV{NVSHMEM_HOME})
        endif()
        # Search for NVSHMEM header
        find_path(NVSHMEM_INCLUDE_DIR
            NAMES nvshmem.h
            HINTS ${NVSHMEM_HOME} $ENV{NVSHMEM_HOME}
            PATHS /usr /usr/local
            PATH_SUFFIXES include include/nvshmem)
        # Search for NVSHMEM host library
        find_library(NVSHMEM_HOST_LIB
            NAMES nvshmem_host
            HINTS ${NVSHMEM_HOME} $ENV{NVSHMEM_HOME}
            PATHS /usr /usr/local
            PATH_SUFFIXES lib lib64)
        # Search for NVSHMEM device library
        find_library(NVSHMEM_DEVICE_LIB
            NAMES nvshmem_device
            HINTS ${NVSHMEM_HOME} $ENV{NVSHMEM_HOME}
            PATHS /usr /usr/local
            PATH_SUFFIXES lib lib64)
        if(NOT NVSHMEM_INCLUDE_DIR OR NOT NVSHMEM_HOST_LIB OR NOT NVSHMEM_DEVICE_LIB)
            message(FATAL_ERROR "NVSHMEM was requested but not all components were found. "
                                "NVSHMEM_HOME=${NVSHMEM_HOME}\n"
                                "NVSHMEM_INCLUDE_DIR=${NVSHMEM_INCLUDE_DIR}\n"
                                "NVSHMEM_HOST_LIB=${NVSHMEM_HOST_LIB}\n"
                                "NVSHMEM_DEVICE_LIB=${NVSHMEM_DEVICE_LIB}\n"
                                "Set NVSHMEM_HOME or specify include/lib paths.")
        endif()
        target_include_directories(${exename}-cuda PUBLIC ${NVSHMEM_INCLUDE_DIR})
        target_link_libraries(${exename}-cuda PRIVATE ${NVSHMEM_HOST_LIB} ${NVSHMEM_DEVICE_LIB})
        set_target_properties(${exename}-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()
    enableGpuDirect(${exename}-cuda)
    install(TARGETS ${exename}-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# elseif(CMAKE_HIP_COMPILER)
#     add_executable(${exename}-hip power_spectrum.cpp)
#     target_compile_definitions(${exename}-hip PRIVATE USE_CUDA THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
#     target_link_libraries(${exename}-hip PRIVATE cstone_gpu io OpenMP::OpenMP_CXX
#         ${MPI_CXX_LIBRARIES} hip::host)
#     set_target_properties(${exename}-hip PROPERTIES LINKER_LANGUAGE CXX)
#     enableGpuDirect(${exename}-hip)
#     install(TARGETS ${exename}-hip RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
