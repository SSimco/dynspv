# SPDX-License-Identifier: MPL-2.0

find_package(GTest REQUIRED)

enable_testing()

add_executable(
  generator_tests
  generator_tests.cpp
)

target_link_libraries(
  generator_tests
  dynspv
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(generator_tests)
