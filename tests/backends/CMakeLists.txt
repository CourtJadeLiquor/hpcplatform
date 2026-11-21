# Backend-level suites: CUDA/OpenCV copy, lifecycle and cross-backend parity.

if(MYCV_WITH_CUDA)
  add_executable(test_cuda_backend test_cuda_backend.cpp)
  target_link_libraries(test_cuda_backend PRIVATE mycv)
  target_include_directories(test_cuda_backend PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
  add_test(NAME test_cuda_backend COMMAND test_cuda_backend)
endif()

if(MYCV_WITH_OPENCV)
  add_executable(test_opencv_backend test_opencv_backend.cpp)
  target_link_libraries(test_opencv_backend PRIVATE mycv)
  target_include_directories(test_opencv_backend PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
  add_test(NAME test_opencv_backend COMMAND test_opencv_backend)
endif()
