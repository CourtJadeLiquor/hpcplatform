# Core-layer suites: shape/tensor/views/memory/backend/copy semantics.
# Each file is one GoogleTest executable registered with CTest.

set(hpcplatform_CORE_TEST_SOURCES
  test_backend.cpp
  test_copy.cpp
  test_enum_names.cpp
  test_memory.cpp
  test_operator_stub.cpp
  test_shape.cpp
  test_tensor.cpp
  test_tensor_views.cpp)

foreach(test_source IN LISTS hpcplatform_CORE_TEST_SOURCES)
  get_filename_component(test_name ${test_source} NAME_WE)
  hpcplatform_add_test(${test_name} ${test_source})
endforeach()
