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
  src/core/copy.cpp
  src/core/dtype.cpp
  src/core/error.cpp
  src/core/shape.cpp
  src/core/tensor.cpp
  src/backends/factory.cpp
  src/backends/cpu/cpu_backend.cpp
  src/backends/cpu/cpu_memory_manager.cpp
  src/backends/npu/npu_backend.cpp
  src/framework/op_registry.cpp
  src/ops/register_ops.cpp
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
# Enabled explicitly at configure time; operator folders compile their own
# OpenMP kernel translation units.
if(MYCV_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(mycv PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(mycv PUBLIC MYCV_WITH_OPENMP=1)
endif()

if(MYCV_ENABLE_TBB)
  find_package(TBB REQUIRED)
  target_link_libraries(mycv PUBLIC TBB::tbb)
  target_compile_definitions(mycv PUBLIC MYCV_WITH_TBB=1)
endif()

# --- SIMD capability ------------------------------------------------------
# No -march=native: each SIMD kernel translation unit gets its own flag inside
# its operator folder so the resulting static library stays portable.
if(MYCV_ENABLE_SIMD)
  target_compile_definitions(mycv PRIVATE MYCV_WITH_SIMD_GENERIC=1)
  # Capability probes only: operator folders gate their own -mavx2 object
  # libraries on MYCV_COMPILER_HAS_AVX2 and define MYCV_ARCH_AVX2 themselves.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" MYCV_COMPILER_HAS_AVX2)
  endif()
endif()

# --- Optional backends ----------------------------------------------------
if(MYCV_WITH_OPENCV)
  find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc)
  target_sources(mycv PRIVATE
    src/backends/opencv/opencv_backend.cpp
    src/backends/opencv/opencv_convert.cpp
    src/backends/opencv/opencv_memory_manager.cpp
    src/backends/opencv/opencv_tensor_access.cpp)
  target_include_directories(mycv SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(mycv PUBLIC ${OpenCV_LIBS})
  target_compile_definitions(mycv PUBLIC MYCV_WITH_OPENCV=1)
endif()

if(MYCV_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(mycv PRIVATE
    src/backends/cuda/cuda_backend.cpp
    src/backends/cuda/cuda_copy.cpp
    src/backends/cuda/cuda_memory_manager.cpp
    src/backends/cuda/cuda_stream.cpp)
  target_link_libraries(mycv PUBLIC CUDA::cudart)
  target_compile_definitions(mycv PUBLIC MYCV_WITH_CUDA=1)
  target_compile_options(mycv PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

# Operator folders are self-contained (sources, optional-backend files,
# per-op tests/benchmarks); each one is mounted with a single line.
# enable_testing() must run before their add_test() calls register.
if(MYCV_BUILD_TESTS)
  enable_testing()
endif()
add_subdirectory(src/ops)

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
