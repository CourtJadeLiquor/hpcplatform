# Backend-level suites: CUDA/OpenCV copy, lifecycle and cross-backend parity.

if(hpcplatform_WITH_CUDA)
  hpcplatform_add_test(test_cuda_backend test_cuda_backend.cpp)
endif()

if(hpcplatform_WITH_OPENCV)
  hpcplatform_add_test(test_opencv_backend test_opencv_backend.cpp)
endif()
