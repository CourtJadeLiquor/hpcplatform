# Operator performance evaluation test (docs/architecture.md §6.4). Profiling
# is enabled only for this target so the library keeps its zero-cost default.
add_executable(test_operator_perf test_operator_perf.cpp)
target_link_libraries(test_operator_perf PRIVATE mycv)
target_include_directories(test_operator_perf PRIVATE
  ${CMAKE_SOURCE_DIR}/tests
  ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(test_operator_perf PRIVATE MYCV_ENABLE_PROFILING=1)
add_test(NAME test_operator_perf COMMAND test_operator_perf)
