if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.22)
    project(marketplace CXX)
    include(CTest)
endif()

add_library(marketplace INTERFACE)
target_include_directories(marketplace INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(marketplace INTERFACE cxx_std_20)

if(BUILD_TESTING)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../main/cmake")
    include(setup_GTest)

    add_executable(marketplace_order_book_tests test/order_book_tests.cpp)
    target_compile_options(marketplace_order_book_tests PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
    target_link_libraries(marketplace_order_book_tests PRIVATE marketplace GTest::gtest_main)
    add_test(NAME MarketplaceOrderBook COMMAND marketplace_order_book_tests)
endif()
