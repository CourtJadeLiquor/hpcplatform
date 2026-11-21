# Self-contained build script of the gaussian_blur operator
# (docs/architecture.md §6.1). Everything is gated by the same compile-time
# switches as the root project; optional-backend implementations live in this
# folder. The AVX2 translation unit gets its own -mavx2 flag so the static
# library stays portable (no -march=native).

target_sources(mycv PRIVATE
  gaussian_blur.cpp
  gaussian_blur_cpu_naive.cpp)

if(MYCV_ENABLE_OPENMP)
  target_sources(mycv PRIVATE gaussian_blur_cpu_omp.cpp)
endif()

if(MYCV_ENABLE_TBB)
  target_sources(mycv PRIVATE gaussian_blur_cpu_tbb.cpp)
endif()

if(MYCV_ENABLE_SIMD)
  target_sources(mycv PRIVATE gaussian_blur_cpu_simd.cpp)
endif()

if(MYCV_COMPILER_HAS_AVX2)
  # Feature macro on mycv itself: registration lives in gaussian_blur.cpp
  # (part of mycv), and it must see MYCV_ARCH_AVX2 to register the AVX2 kernel.
  # The -mavx2 flag stays confined to the object library below.
  target_compile_definitions(mycv PRIVATE MYCV_ARCH_AVX2=1)
  # Per-source COMPILE_OPTIONS are directory-scoped and cannot reach the
  # mycv target (created in the parent directory), so the AVX2 unit gets its
  # own object library carrying the flag. Keeps the static library portable.
  add_library(mycv_gaussian_blur_avx2 OBJECT gaussian_blur_cpu_avx2.cpp)
  target_compile_options(mycv_gaussian_blur_avx2 PRIVATE -mavx2)
  target_compile_definitions(mycv_gaussian_blur_avx2 PRIVATE MYCV_ARCH_AVX2=1)
  target_include_directories(mycv_gaussian_blur_avx2 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src)
  # Embed the objects directly instead of target_link_libraries: linking a
  # helper target into mycv would force it into the install(EXPORT) set.
  target_sources(mycv PRIVATE $<TARGET_OBJECTS:mycv_gaussian_blur_avx2>)
endif()

if(MYCV_WITH_OPENCV)
  target_sources(mycv PRIVATE gaussian_blur_opencv.cpp)
endif()

if(MYCV_WITH_CUDA)
  target_sources(mycv PRIVATE gaussian_blur_cuda.cu)
endif()

if(MYCV_BUILD_TESTS)
  add_executable(test_gaussian_blur tests/test_gaussian_blur.cpp)
  target_link_libraries(test_gaussian_blur PRIVATE mycv)
  target_include_directories(test_gaussian_blur PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
  add_test(NAME test_gaussian_blur COMMAND test_gaussian_blur)
endif()

if(MYCV_BUILD_BENCHMARKS)
  add_executable(gaussian_blur_benchmark benchmark/gaussian_blur_benchmark.cpp)
  target_link_libraries(gaussian_blur_benchmark PRIVATE mycv)
  target_include_directories(gaussian_blur_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_SOURCE_DIR}/src)
endif()
