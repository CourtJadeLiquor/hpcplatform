# Operator performance evaluation test (docs/architecture.md §8.2). Profiling
# is enabled only for this target so the library keeps its zero-cost default.
hpcplatform_add_test(test_operator_perf test_operator_perf.cpp)
target_compile_definitions(test_operator_perf PRIVATE hpcplatform_ENABLE_PROFILING=1)
