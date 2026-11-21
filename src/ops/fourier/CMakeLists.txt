# Self-contained build script of the fourier operator (docs/architecture.md
# §6.1). CPU levels are naive + OpenMP + TBB; the CUDA device kernel lives in
# fourier_cuda.cu and is compiled only when MYCV_WITH_CUDA is on. The operator
# also ships its own tests and benchmark.

target_sources(mycv PRIVATE
  fourier.cpp
  fourier_cpu_naive.cpp)

if(MYCV_ENABLE_OPENMP)
  target_sources(mycv PRIVATE fourier_cpu_omp.cpp)
endif()

if(MYCV_ENABLE_TBB)
  target_sources(mycv PRIVATE fourier_cpu_tbb.cpp)
endif()

if(MYCV_WITH_CUDA)
  target_sources(mycv PRIVATE fourier_cuda.cu)
endif()

if(MYCV_BUILD_TESTS)
  add_executable(test_fourier tests/test_fourier.cpp)
  target_link_libraries(test_fourier PRIVATE mycv)
  target_include_directories(test_fourier PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
  add_test(NAME test_fourier COMMAND test_fourier)
endif()

if(MYCV_BUILD_BENCHMARKS)
  add_executable(fourier_benchmark benchmark/fourier_benchmark.cpp)
  target_link_libraries(fourier_benchmark PRIVATE mycv)
  target_include_directories(fourier_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
endif()
