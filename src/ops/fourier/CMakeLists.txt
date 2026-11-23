# Self-contained build script of the fourier operator (docs/architecture.md
# §6.1). CPU levels are naive + OpenMP + TBB; the CUDA device kernel lives in
# fourier_cuda.cu and is compiled only when hpcplatform_WITH_CUDA is on. The operator
# also ships its own tests and benchmark.

target_sources(hpcplatform PRIVATE
  fourier.cpp
  fourier_cpu_naive.cpp)

if(hpcplatform_ENABLE_OPENMP)
  target_sources(hpcplatform PRIVATE fourier_cpu_omp.cpp)
endif()

if(hpcplatform_ENABLE_TBB)
  target_sources(hpcplatform PRIVATE fourier_cpu_tbb.cpp)
endif()

if(hpcplatform_WITH_CUDA)
  target_sources(hpcplatform PRIVATE fourier_cuda.cu)
endif()

if(hpcplatform_WITH_YAML)
  target_sources(hpcplatform PRIVATE fourier_operator.cpp)
endif()

if(hpcplatform_BUILD_TESTS)
  hpcplatform_add_test(test_fourier tests/test_fourier.cpp)
endif()

if(hpcplatform_BUILD_BENCHMARKS)
  add_executable(fourier_benchmark benchmark/fourier_benchmark.cpp)
  target_link_libraries(fourier_benchmark PRIVATE hpcplatform)
  target_include_directories(fourier_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
endif()
