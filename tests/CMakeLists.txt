set(MYCV_TEST_SOURCES
  consistency/test_cpu_kernels.cpp
  unit/test_backend.cpp
  unit/test_copy_cast.cpp
  unit/test_memory.cpp
  unit/test_ops_elementwise.cpp
  unit/test_ops_image.cpp
  unit/test_shape.cpp
  unit/test_tensor.cpp
  unit/test_tensor_views.cpp)

foreach(test_source IN LISTS MYCV_TEST_SOURCES)
  get_filename_component(test_name ${test_source} NAME_WE)
  add_executable(${test_name} ${test_source})
  target_link_libraries(${test_name} PRIVATE mycv)
  target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

if(MYCV_WITH_CUDA)
  add_executable(test_cuda_backend consistency/test_cuda_backend.cpp)
  target_link_libraries(test_cuda_backend PRIVATE mycv)
  target_include_directories(test_cuda_backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME test_cuda_backend COMMAND test_cuda_backend)
endif()

if(MYCV_WITH_OPENCV)
  add_executable(test_opencv_backend consistency/test_opencv_backend.cpp)
  target_link_libraries(test_opencv_backend PRIVATE mycv)
  target_include_directories(test_opencv_backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(NAME test_opencv_backend COMMAND test_opencv_backend)
endif()
