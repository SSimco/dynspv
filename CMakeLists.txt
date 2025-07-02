# SPDX-License-Identifier: MPL-2.0
cmake_minimum_required(VERSION 3.10.0)
project(dynspv VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

find_package(SPIRV-Tools REQUIRED)

add_library(dynspv INTERFACE)
target_include_directories(dynspv INTERFACE "include")
target_link_libraries(dynspv INTERFACE SPIRV-Tools)

option(DYNSPV_ENABLE_TESTS "Enable tests" OFF)

if(DYNSPV_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()