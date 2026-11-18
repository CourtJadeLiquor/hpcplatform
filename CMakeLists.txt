cmake_minimum_required(VERSION 3.20)

project(mycv
  VERSION 0.1.0
  DESCRIPTION "Cross-platform operator library for research (CPU/OpenCV/CUDA/NPU)"
  LANGUAGES CXX)

option(MYCV_BUILD_TESTS "Build unit tests" ON)
option(MYCV_BUILD_SAMPLES "Build samples" ON)
option(MYCV_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(MYCV_ENABLE_OPENMP "Build the OpenMP CPU kernel path" ON)
option(MYCV_ENABLE_TBB "Build the oneTBB CPU kernel path" OFF)
option(MYCV_ENABLE_SIMD "Enable portable and instruction-set SIMD kernels" ON)
option(MYCV_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(MYCV_WITH_OPENCV "Build the OpenCV backend" OFF)
option(MYCV_WITH_CUDA "Build the CUDA backend" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MYCV_WITH_CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
      "MYCV_WITH_CUDA requires an explicit CMAKE_CUDA_ARCHITECTURES value (for example 89)")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

add_library(mycv STATIC
  src/core/context.cpp
  src/core/dtype.cpp
  src/core/error.cpp
  src/core/shape.cpp
  src/core/tensor.cpp
  src/backends/backend_copy.cpp
  src/backends/backend_factory.cpp
  src/backends/cpu/cpu_backend.cpp
  src/backends/cpu/cpu_memory_manager.cpp
  src/backends/npu/npu_backend.cpp
  src/ops/op_registry.cpp
  src/ops/add/add.cpp
  src/ops/add/add_cpu_naive.cpp
  src/ops/mul/mul.cpp
  src/ops/mul/mul_cpu_naive.cpp
  src/ops/cast/cast.cpp
  src/ops/cast/cast_cpu_naive.cpp
  src/ops/normalize/normalize.cpp
  src/ops/normalize/normalize_cpu_naive.cpp
  src/ops/resize/resize.cpp
  src/ops/resize/resize_cpu_naive.cpp
  src/ops/crop/crop.cpp
  src/ops/crop/crop_cpu_naive.cpp
  src/ops/transpose/transpose.cpp
  src/ops/transpose/transpose_cpu_naive.cpp
  src/ops/warp_affine/warp_affine.cpp
  src/ops/warp_affine/warp_affine_cpu_naive.cpp
)

target_include_directories(mycv
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(MSVC)
  target_compile_options(mycv PRIVATE /W4)
else()
  target_compile_options(mycv PRIVATE
    -Wall
    -Wextra
    $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wpedantic>)
endif()

if(MYCV_ENABLE_SANITIZERS)
  if(MSVC)
    message(FATAL_ERROR "MYCV_ENABLE_SANITIZERS currently requires GCC or Clang")
  endif()
  target_compile_options(mycv PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(mycv PUBLIC -fsanitize=address,undefined)
endif()

# --- OpenMP ---------------------------------------------------------------
# Enabled explicitly at configure time; kernels that need it are compiled in P3.
if(MYCV_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(mycv PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(mycv PUBLIC MYCV_WITH_OPENMP=1)
  target_sources(mycv PRIVATE
    src/ops/add/add_cpu_omp.cpp
    src/ops/mul/mul_cpu_omp.cpp
    src/ops/normalize/normalize_cpu_omp.cpp
    src/ops/resize/resize_cpu_omp.cpp
    src/ops/transpose/transpose_cpu_omp.cpp)
endif()

if(MYCV_ENABLE_TBB)
  find_package(TBB REQUIRED)
  target_link_libraries(mycv PUBLIC TBB::tbb)
  target_compile_definitions(mycv PUBLIC MYCV_WITH_TBB=1)
  target_sources(mycv PRIVATE
    src/ops/add/add_cpu_tbb.cpp
    src/ops/mul/mul_cpu_tbb.cpp
    src/ops/normalize/normalize_cpu_tbb.cpp
    src/ops/resize/resize_cpu_tbb.cpp
    src/ops/transpose/transpose_cpu_tbb.cpp)
endif()

# --- SIMD capability ------------------------------------------------------
# No -march=native: each SIMD kernel translation unit gets its own flag in P3 so
# the resulting static library stays portable.
if(MYCV_ENABLE_SIMD)
  target_compile_definitions(mycv PRIVATE MYCV_WITH_SIMD_GENERIC=1)
  target_sources(mycv PRIVATE
    src/ops/add/add_cpu_simd.cpp
    src/ops/mul/mul_cpu_simd.cpp
    src/ops/cast/cast_cpu_simd.cpp
    src/ops/normalize/normalize_cpu_simd.cpp)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" MYCV_COMPILER_HAS_AVX2)
    if(MYCV_COMPILER_HAS_AVX2)
      target_compile_definitions(mycv PRIVATE MYCV_ARCH_AVX2=1)
      set(MYCV_AVX2_SOURCES
        src/ops/add/add_cpu_avx2.cpp
        src/ops/mul/mul_cpu_avx2.cpp
        src/ops/cast/cast_cpu_avx2.cpp
        src/ops/normalize/normalize_cpu_avx2.cpp)
      target_sources(mycv PRIVATE ${MYCV_AVX2_SOURCES})
      set_source_files_properties(${MYCV_AVX2_SOURCES} PROPERTIES
        COMPILE_OPTIONS "-mavx2")
    endif()
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    target_compile_definitions(mycv PRIVATE MYCV_ARCH_NEON=1)
  endif()
endif()

# --- Optional backends ----------------------------------------------------
if(MYCV_WITH_OPENCV)
  find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc)
  target_sources(mycv PRIVATE
    src/backends/opencv/opencv_backend.cpp
    src/backends/opencv/opencv_convert.cpp
    src/backends/opencv/opencv_kernels.cpp
    src/backends/opencv/opencv_memory_manager.cpp)
  target_include_directories(mycv SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(mycv PUBLIC ${OpenCV_LIBS})
  target_compile_definitions(mycv PUBLIC MYCV_WITH_OPENCV=1)
endif()

if(MYCV_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(mycv PRIVATE
    src/backends/cuda/cuda_backend.cpp
    src/backends/cuda/cuda_copy.cpp
    src/backends/cuda/cuda_kernels.cu
    src/backends/cuda/cuda_memory_manager.cpp
    src/backends/cuda/cuda_stream.cpp)
  target_link_libraries(mycv PUBLIC CUDA::cudart)
  target_compile_definitions(mycv PUBLIC MYCV_WITH_CUDA=1)
  target_compile_options(mycv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

include(GNUInstallDirs)
install(TARGETS mycv EXPORT mycvTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT mycvTargets
  FILE mycvTargets.cmake
  NAMESPACE mycv::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mycv)

if(MYCV_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(MYCV_BUILD_SAMPLES)
  add_subdirectory(samples)
endif()

if(MYCV_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
