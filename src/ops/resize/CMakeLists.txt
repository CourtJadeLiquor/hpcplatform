# Self-contained build script of the resize operator (docs/architecture.md
# §6.1). Everything is gated by the same compile-time switches as the root
# project; optional-backend implementations live in this folder.

target_sources(hpcplatform PRIVATE
  resize.cpp
  resize_cpu_naive.cpp)

if(hpcplatform_ENABLE_OPENMP)
  target_sources(hpcplatform PRIVATE resize_cpu_omp.cpp)
endif()

if(hpcplatform_ENABLE_TBB)
  target_sources(hpcplatform PRIVATE resize_cpu_tbb.cpp)
endif()

if(hpcplatform_WITH_OPENCV)
  target_sources(hpcplatform PRIVATE resize_opencv.cpp)
endif()

if(hpcplatform_WITH_CUDA)
  target_sources(hpcplatform PRIVATE resize_cuda.cu)
endif()

if(hpcplatform_WITH_YAML)
  target_sources(hpcplatform PRIVATE resize_operator.cpp)
endif()

if(hpcplatform_BUILD_TESTS)
  hpcplatform_add_test(test_resize tests/test_resize.cpp)
endif()

if(hpcplatform_BUILD_BENCHMARKS)
  add_executable(resize_benchmark benchmark/resize_benchmark.cpp)
  target_link_libraries(resize_benchmark PRIVATE hpcplatform)
  target_include_directories(resize_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
endif()
