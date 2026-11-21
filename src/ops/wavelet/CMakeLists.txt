# Self-contained build script of the wavelet operator (docs/architecture.md
# §6.1). CPU levels are naive + OpenMP + TBB; the CUDA device kernel lives in
# wavelet_cuda.cu and is compiled only when MYCV_WITH_CUDA is on. The operator
# also ships its own tests and benchmark.

target_sources(mycv PRIVATE
  wavelet.cpp
  wavelet_cpu_naive.cpp)

if(MYCV_ENABLE_OPENMP)
  target_sources(mycv PRIVATE wavelet_cpu_omp.cpp)
endif()

if(MYCV_ENABLE_TBB)
  target_sources(mycv PRIVATE wavelet_cpu_tbb.cpp)
endif()

if(MYCV_WITH_CUDA)
  target_sources(mycv PRIVATE wavelet_cuda.cu)
endif()

if(MYCV_BUILD_TESTS)
  add_executable(test_wavelet tests/test_wavelet.cpp)
  target_link_libraries(test_wavelet PRIVATE mycv)
  target_include_directories(test_wavelet PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
  add_test(NAME test_wavelet COMMAND test_wavelet)
endif()

if(MYCV_BUILD_BENCHMARKS)
  add_executable(wavelet_benchmark benchmark/wavelet_benchmark.cpp)
  target_link_libraries(wavelet_benchmark PRIVATE mycv)
  target_include_directories(wavelet_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
endif()
