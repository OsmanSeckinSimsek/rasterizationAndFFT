
set(UNIT_TESTS
    test_main.cpp)

if(SPH_EXA_WITH_H5PART)
    set(exename frontend_units)
    add_executable(${exename} ${UNIT_TESTS})
    target_compile_options(${exename} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)

    target_include_directories(${exename} PRIVATE ${MPI_CXX_INCLUDE_PATH} ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/main/src)
    target_link_libraries(${exename} PRIVATE ${MPI_CXX_LIBRARIES} GTest::gtest_main)
    enableH5Part(${exename})
    add_test(NAME FrontendUnits COMMAND ${exename})
endif()

if(CMAKE_CUDA_COMPILER AND SPH_EXA_WITH_H5PART)
    set(testname frontend_units_cuda)
    add_executable(${testname}
        test_main.cpp)
    target_include_directories(${testname} PRIVATE
        ${CSTONE_DIR}
        ${PROJECT_SOURCE_DIR}/main/src)
    target_link_libraries(${testname} PUBLIC cstone_gpu OpenMP::OpenMP_CXX CUDA::cudart GTest::gtest_main)
    add_test(NAME FrontendUnitsCuda COMMAND ${exename})
    unset(testname)
endif()
