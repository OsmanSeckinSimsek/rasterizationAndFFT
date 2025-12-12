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
        message(STATUS "Searching for NVSHMEM with NVSHMEM_HOME=${NVSHMEM_HOME}")
        # Search for NVSHMEM header - try multiple possible locations and names
        find_path(NVSHMEM_INCLUDE_DIR
            NAMES nvshmem.h nvshmem/nvshmem.h
            HINTS ${NVSHMEM_HOME} $ENV{NVSHMEM_HOME}
            PATHS /usr /usr/local /opt/nvshmem
            PATH_SUFFIXES include include/nvshmem)
        # Search for NVSHMEM host library - try multiple possible names
        find_library(NVSHMEM_HOST_LIB
            NAMES nvshmem_host libnvshmem_host nvshmem_host_2.0 nvshmem_host_2.1 nvshmem_host_2.2
            HINTS ${NVSHMEM_HOME} $ENV{NVSHMEM_HOME}
            PATHS /usr /usr/local /opt/nvshmem
            PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
        # Search for NVSHMEM device library - try multiple possible names
        find_library(NVSHMEM_DEVICE_LIB
            NAMES nvshmem_device libnvshmem_device nvshmem_device_2.0 nvshmem_device_2.1 nvshmem_device_2.2
            HINTS ${NVSHMEM_HOME} $ENV{NVSHMEM_HOME}
            PATHS /usr /usr/local /opt/nvshmem
            PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
        message(STATUS "NVSHMEM search results:")
        message(STATUS "  NVSHMEM_INCLUDE_DIR=${NVSHMEM_INCLUDE_DIR}")
        message(STATUS "  NVSHMEM_HOST_LIB=${NVSHMEM_HOST_LIB}")
        message(STATUS "  NVSHMEM_DEVICE_LIB=${NVSHMEM_DEVICE_LIB}")
        
        # If not found, try to find any nvshmem libraries as fallback
        if(NOT NVSHMEM_HOST_LIB)
            foreach(search_path ${NVSHMEM_HOME} /usr /usr/local)
                file(GLOB NVSHMEM_HOST_CANDIDATES 
                    "${search_path}/lib/libnvshmem*host*"
                    "${search_path}/lib64/libnvshmem*host*"
                    "${search_path}/lib/x86_64-linux-gnu/libnvshmem*host*")
                if(NVSHMEM_HOST_CANDIDATES)
                    list(GET NVSHMEM_HOST_CANDIDATES 0 NVSHMEM_HOST_LIB)
                    message(STATUS "  Found NVSHMEM_HOST_LIB via fallback: ${NVSHMEM_HOST_LIB}")
                    break()
                endif()
            endforeach()
        endif()
        
        if(NOT NVSHMEM_DEVICE_LIB)
            foreach(search_path ${NVSHMEM_HOME} /usr /usr/local)
                file(GLOB NVSHMEM_DEVICE_CANDIDATES 
                    "${search_path}/lib/libnvshmem*device*"
                    "${search_path}/lib64/libnvshmem*device*"
                    "${search_path}/lib/x86_64-linux-gnu/libnvshmem*device*")
                if(NVSHMEM_DEVICE_CANDIDATES)
                    list(GET NVSHMEM_DEVICE_CANDIDATES 0 NVSHMEM_DEVICE_LIB)
                    message(STATUS "  Found NVSHMEM_DEVICE_LIB via fallback: ${NVSHMEM_DEVICE_LIB}")
                    break()
                endif()
            endforeach()
        endif()
        
        if(NOT NVSHMEM_INCLUDE_DIR)
            foreach(search_path ${NVSHMEM_HOME} /usr /usr/local)
                if(EXISTS "${search_path}/include/nvshmem.h")
                    set(NVSHMEM_INCLUDE_DIR "${search_path}/include")
                    message(STATUS "  Found NVSHMEM_INCLUDE_DIR via fallback: ${NVSHMEM_INCLUDE_DIR}")
                    break()
                elseif(EXISTS "${search_path}/include/nvshmem/nvshmem.h")
                    set(NVSHMEM_INCLUDE_DIR "${search_path}/include/nvshmem")
                    message(STATUS "  Found NVSHMEM_INCLUDE_DIR via fallback: ${NVSHMEM_INCLUDE_DIR}")
                    break()
                endif()
            endforeach()
        endif()
        
        if(NOT NVSHMEM_INCLUDE_DIR OR NOT NVSHMEM_HOST_LIB OR NOT NVSHMEM_DEVICE_LIB)
            message(FATAL_ERROR "NVSHMEM was requested but not all components were found. "
                                "NVSHMEM_HOME=${NVSHMEM_HOME}\n"
                                "NVSHMEM_INCLUDE_DIR=${NVSHMEM_INCLUDE_DIR}\n"
                                "NVSHMEM_HOST_LIB=${NVSHMEM_HOST_LIB}\n"
                                "NVSHMEM_DEVICE_LIB=${NVSHMEM_DEVICE_LIB}\n"
                                "Please check the debug output above to see where NVSHMEM files are located.\n"
                                "Set NVSHMEM_HOME or specify include/lib paths.")
        endif()
        target_include_directories(${exename}-cuda PUBLIC ${NVSHMEM_INCLUDE_DIR})
        target_link_libraries(${exename}-cuda PRIVATE ${NVSHMEM_HOST_LIB} ${NVSHMEM_DEVICE_LIB})
        set_target_properties(${exename}-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        # Set RPATH so the binary can find NVSHMEM libraries at runtime
        get_filename_component(NVSHMEM_LIB_DIR ${NVSHMEM_HOST_LIB} DIRECTORY)
        set_target_properties(${exename}-cuda PROPERTIES
            BUILD_RPATH "${NVSHMEM_LIB_DIR}"
            INSTALL_RPATH "${NVSHMEM_LIB_DIR}"
            BUILD_RPATH_USE_ORIGIN TRUE
            INSTALL_RPATH_USE_LINK_PATH TRUE)
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
