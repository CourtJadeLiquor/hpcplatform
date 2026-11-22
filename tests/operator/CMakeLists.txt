# Config-driven operator suite. Only built when MYCV_WITH_YAML is on, because
# the real LoadOperators implementation (and these tests) require yaml-cpp.
# AC-6 / the default-build stub is covered by tests/core/test_operator_stub.cpp.
add_executable(test_load_operators test_load_operators.cpp)
target_link_libraries(test_load_operators PRIVATE mycv)
target_include_directories(test_load_operators PRIVATE
  ${CMAKE_SOURCE_DIR}/tests
  ${CMAKE_SOURCE_DIR}/src)
add_test(NAME test_load_operators COMMAND test_load_operators)
