set(CSTONE_DIR ${PROJECT_SOURCE_DIR}/domain/include)

add_subdirectory(src)

if(BUILD_TESTING)
    add_subdirectory(test)
endif(BUILD_TESTING)
