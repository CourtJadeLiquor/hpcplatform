# Core-layer suites: shape/tensor/views/memory/backend/copy semantics.

set(MYCV_CORE_TEST_SOURCES
  test_backend.cpp
  test_copy.cpp
  test_enum_names.cpp
  test_memory.cpp
  test_operator_stub.cpp
  test_shape.cpp
  test_tensor.cpp
  test_tensor_views.cpp)

foreach(test_source IN LISTS MYCV_CORE_TEST_SOURCES)
  get_filename_component(test_name ${test_source} NAME_WE)
  add_executable(${test_name} ${test_source})
  target_link_libraries(${test_name} PRIVATE mycv)
  target_include_directories(${test_name} PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
