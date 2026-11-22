# Self-contained build script of the laplacian_pyramid operator
# (docs/architecture.md §6.1/§11). The operator composes the resize and
# gaussian_blur kernels, so it has no SIMD/AVX2 units of its own: only the
# host composition, the CUDA subtract kernel and the per-op tests/benchmark.

target_sources(mycv PRIVATE
  laplacian_pyramid.cpp
  laplacian_pyramid_host.cpp)

if(MYCV_WITH_CUDA)
  target_sources(mycv PRIVATE laplacian_pyramid_cuda.cu)
endif()

if(MYCV_WITH_YAML)
  target_sources(mycv PRIVATE laplacian_pyramid_operator.cpp)
endif()

if(MYCV_BUILD_TESTS)
  add_executable(test_laplacian_pyramid tests/test_laplacian_pyramid.cpp)
  target_link_libraries(test_laplacian_pyramid PRIVATE mycv)
  target_include_directories(test_laplacian_pyramid PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
  add_test(NAME test_laplacian_pyramid COMMAND test_laplacian_pyramid)
endif()

if(MYCV_BUILD_BENCHMARKS)
  add_executable(laplacian_pyramid_benchmark
    benchmark/laplacian_pyramid_benchmark.cpp)
  target_link_libraries(laplacian_pyramid_benchmark PRIVATE mycv)
  target_include_directories(laplacian_pyramid_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
endif()
