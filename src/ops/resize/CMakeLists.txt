# Self-contained build script of the resize operator (docs/architecture.md
# §6.1). Everything is gated by the same compile-time switches as the root
# project; optional-backend implementations live in this folder.

target_sources(mycv PRIVATE
  resize.cpp
  resize_cpu_naive.cpp)

if(MYCV_ENABLE_OPENMP)
  target_sources(mycv PRIVATE resize_cpu_omp.cpp)
endif()

if(MYCV_ENABLE_TBB)
  target_sources(mycv PRIVATE resize_cpu_tbb.cpp)
endif()

if(MYCV_WITH_OPENCV)
  target_sources(mycv PRIVATE resize_opencv.cpp)
endif()

if(MYCV_WITH_CUDA)
  target_sources(mycv PRIVATE resize_cuda.cu)
endif()

if(MYCV_WITH_YAML)
  target_sources(mycv PRIVATE resize_operator.cpp)
endif()

if(MYCV_BUILD_TESTS)
  add_executable(test_resize tests/test_resize.cpp)
  target_link_libraries(test_resize PRIVATE mycv)
  target_include_directories(test_resize PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
  add_test(NAME test_resize COMMAND test_resize)
endif()

if(MYCV_BUILD_BENCHMARKS)
  add_executable(resize_benchmark benchmark/resize_benchmark.cpp)
  target_link_libraries(resize_benchmark PRIVATE mycv)
  target_include_directories(resize_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
endif()
