# Self-contained build script of the gaussian_blur operator
# (docs/architecture.md §6.1). Everything is gated by the same compile-time
# switches as the root project; optional-backend implementations live in this
# folder. The AVX2 translation unit gets its own -mavx2 flag so the static
# library stays portable (no -march=native).

target_sources(hpcplatform PRIVATE
  gaussian_blur.cpp
  gaussian_blur_cpu_naive.cpp)

if(hpcplatform_ENABLE_OPENMP)
  target_sources(hpcplatform PRIVATE gaussian_blur_cpu_omp.cpp)
endif()

if(hpcplatform_ENABLE_TBB)
  target_sources(hpcplatform PRIVATE gaussian_blur_cpu_tbb.cpp)
endif()

if(hpcplatform_ENABLE_SIMD)
  target_sources(hpcplatform PRIVATE gaussian_blur_cpu_simd.cpp)
endif()

if(hpcplatform_COMPILER_HAS_AVX2)
  # Feature macro on hpcplatform itself: registration lives in gaussian_blur.cpp
  # (part of hpcplatform), and it must see hpcplatform_ARCH_AVX2 to register the AVX2 kernel.
  # The -mavx2 flag stays confined to the object library below.
  target_compile_definitions(hpcplatform PRIVATE hpcplatform_ARCH_AVX2=1)
  # Per-source COMPILE_OPTIONS are directory-scoped and cannot reach the
  # hpcplatform target (created in the parent directory), so the AVX2 unit gets its
  # own object library carrying the flag. Keeps the static library portable.
  add_library(hpcplatform_gaussian_blur_avx2 OBJECT gaussian_blur_cpu_avx2.cpp)
  target_compile_options(hpcplatform_gaussian_blur_avx2 PRIVATE -mavx2)
  target_compile_definitions(hpcplatform_gaussian_blur_avx2 PRIVATE hpcplatform_ARCH_AVX2=1)
  target_include_directories(hpcplatform_gaussian_blur_avx2 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src)
  # Embed the objects directly instead of target_link_libraries: linking a
  # helper target into hpcplatform would force it into the install(EXPORT) set.
  target_sources(hpcplatform PRIVATE $<TARGET_OBJECTS:hpcplatform_gaussian_blur_avx2>)
endif()

if(hpcplatform_WITH_OPENCV)
  target_sources(hpcplatform PRIVATE gaussian_blur_opencv.cpp)
endif()

if(hpcplatform_WITH_CUDA)
  target_sources(hpcplatform PRIVATE gaussian_blur_cuda.cu)
endif()

if(hpcplatform_WITH_YAML)
  target_sources(hpcplatform PRIVATE gaussian_blur_operator.cpp)
endif()

if(hpcplatform_BUILD_TESTS)
  hpcplatform_add_test(test_gaussian_blur tests/test_gaussian_blur.cpp)
endif()

if(hpcplatform_BUILD_BENCHMARKS)
  add_executable(gaussian_blur_benchmark benchmark/gaussian_blur_benchmark.cpp)
  target_link_libraries(gaussian_blur_benchmark PRIVATE hpcplatform)
  target_include_directories(gaussian_blur_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
endif()
