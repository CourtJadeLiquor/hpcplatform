# Config-driven operator suite. Only built when hpcplatform_WITH_YAML is on, because
# the real LoadOperators implementation (and these tests) require yaml-cpp.
# AC-6 / the default-build stub is covered by tests/core/test_operator_stub.cpp.
hpcplatform_add_test(test_load_operators test_load_operators.cpp)
