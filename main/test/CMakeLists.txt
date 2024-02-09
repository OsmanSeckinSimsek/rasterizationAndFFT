include(rasterization_add_test)

# set(UNIT_TESTS
# test_main.cpp)
find_package(Heffte PATHS $(HEFFTE_PATH))
find_package(FFTW3 PATHS $(FFTW_PATH))

if(FFTW3_FOUND)
    message(STATUS "Found FFTW3: ${FFTW3_INCLUDE_DIRS}")
endif()

function(addMpiTest source exename testname ranks)
    add_executable(${exename} ${source})
    target_include_directories(${exename} PRIVATE ${MPI_CXX_INCLUDE_PATH} ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/main/src ${HEFFTE_PATH}/include ${FFTW3_INCLUDE_DIRS})
    target_compile_options(${exename} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
    target_link_libraries(${exename} PRIVATE ${MPI_CXX_LIBRARIES} GTest::gtest_main Heffte::Heffte)

    if(ENABLE_H5PART)
        enableH5Part(${exename})
    endif()

    rasterization_add_test(${testname} EXECUTABLE ${exename} RANKS ${ranks})
    install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/integration_mpi)
endfunction()

function(addRasterizationMpiTest source exename testname ranks)
    list(APPEND source test_main.cpp)
    addMpiTest("${source}" ${exename} ${testname} ${ranks})
endfunction()

addRasterizationMpiTest(heffte_tests.cpp heffte_mpi HeffteDFT 4)
addRasterizationMpiTest(mesh_tests.cpp mesh_mpi meshTest 4)

# if(ENABLE_H5PART)
# set(exename frontend_units)
# add_executable(${exename} ${UNIT_TESTS})
# target_compile_options(${exename} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)

# target_include_directories(${exename} PRIVATE ${MPI_CXX_INCLUDE_PATH} ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/main/src ${HEFFTE_PATH}/include ${FFTW3_INCLUDE_DIRS})
# target_link_libraries(${exename} PRIVATE ${MPI_CXX_LIBRARIES} GTest::gtest_main Heffte::Heffte)
# enableH5Part(${exename})
# add_test(NAME FrontendUnits COMMAND ${exename})
# endif()
# if(CMAKE_CUDA_COMPILER AND ENABLE_H5PART)
# set(testname frontend_units_cuda)
# add_executable(${testname}
# test_main.cpp)
# target_include_directories(${testname} PRIVATE
# ${CSTONE_DIR}
# ${PROJECT_SOURCE_DIR}/main/src)
# target_link_libraries(${testname} PUBLIC cstone_gpu OpenMP::OpenMP_CXX CUDA::cudart GTest::gtest_main)
# add_test(NAME FrontendUnitsCuda COMMAND ${exename})
# unset(testname)
# endif()
