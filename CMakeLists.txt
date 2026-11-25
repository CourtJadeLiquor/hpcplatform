cmake_minimum_required(VERSION 3.20)

# set(CMAKE_C_COMPILER "clang")
# set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Debug)

project(hpcplatform
  VERSION 0.2.1
  DESCRIPTION "Cross-platform operator library for research (CPU/OpenCV/CUDA/NPU)"
  LANGUAGES CXX)

# Build type: -DCMAKE_BUILD_TYPE wins; when omitted it is inferred from the build
# directory name so the two supported build trees work without extra flags:
#   cmake -S . -B build_debug    -> Debug
#   cmake -S . -B build_release  -> Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  get_filename_component(hpcplatform_BUILD_DIR_NAME "${CMAKE_BINARY_DIR}" NAME)
  if(hpcplatform_BUILD_DIR_NAME MATCHES "[Rr]elease")
    set(hpcplatform_DEFAULT_BUILD_TYPE Release)
  else()
    set(hpcplatform_DEFAULT_BUILD_TYPE Debug)
  endif()
  set(CMAKE_BUILD_TYPE "${hpcplatform_DEFAULT_BUILD_TYPE}" CACHE STRING "Build type" FORCE)
  message(STATUS "hpcplatform: CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} (inferred from build directory '${hpcplatform_BUILD_DIR_NAME}')")
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release)

# Sanitizers are a development-only aid: on for Debug, off for Release.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(hpcplatform_SANITIZERS_DEFAULT ON)
else()
  set(hpcplatform_SANITIZERS_DEFAULT OFF)
endif()

option(hpcplatform_BUILD_TESTS "Build unit tests" ON)
option(hpcplatform_BUILD_SAMPLES "Build samples" ON)
option(hpcplatform_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(hpcplatform_ENABLE_OPENMP "Build the OpenMP CPU kernel path" ON)
option(hpcplatform_ENABLE_TBB "Build the oneTBB CPU kernel path" ON)
option(hpcplatform_ENABLE_SIMD "Enable portable and instruction-set SIMD kernels" ON)
option(hpcplatform_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" ${hpcplatform_SANITIZERS_DEFAULT})
option(hpcplatform_WITH_OPENCV "Build the OpenCV backend" ON)
option(hpcplatform_WITH_CUDA "Build the CUDA backend" ON)
option(hpcplatform_WITH_YAML "Build the YAML config-driven operator loader" ON)

if(hpcplatform_WITH_CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

add_library(hpcplatform STATIC
  src/core/context.cpp
  src/core/copy.cpp
  src/core/error.cpp
  src/core/shape.cpp
  src/core/tensor.cpp
  src/backends/factory.cpp
  src/backends/cpu/cpu_backend.cpp
  src/backends/cpu/cpu_memory_manager.cpp
  src/backends/npu/npu_backend.cpp
  src/framework/op_registry.cpp
  src/framework/op_descriptor.cpp
  src/ops/loader/load_operators.cpp
  src/ops/register_ops.cpp
)

target_include_directories(hpcplatform
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(MSVC)
  target_compile_options(hpcplatform PRIVATE /W4)
else()
  target_compile_options(hpcplatform PRIVATE
    -Wall
    -Wextra
    $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wpedantic>)
endif()

if(hpcplatform_ENABLE_SANITIZERS)
  if(MSVC)
    message(FATAL_ERROR "hpcplatform_ENABLE_SANITIZERS currently requires GCC or Clang")
  endif()
  target_compile_options(hpcplatform PUBLIC
    $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fsanitize=address,undefined>
    $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-fno-omit-frame-pointer>)
  target_link_options(hpcplatform PUBLIC -fsanitize=address,undefined)
endif()

# --- OpenMP ---------------------------------------------------------------
# Enabled explicitly at configure time; operator folders compile their own
# OpenMP kernel translation units.
if(hpcplatform_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(hpcplatform PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(hpcplatform PUBLIC hpcplatform_WITH_OPENMP=1)
endif()

if(hpcplatform_ENABLE_TBB)
  find_package(TBB REQUIRED)
  target_link_libraries(hpcplatform PUBLIC TBB::tbb)
  target_compile_definitions(hpcplatform PUBLIC hpcplatform_WITH_TBB=1)
endif()

# --- SIMD capability ------------------------------------------------------
# No -march=native: each SIMD kernel translation unit gets its own flag inside
# its operator folder so the resulting static library stays portable.
if(hpcplatform_ENABLE_SIMD)
  target_compile_definitions(hpcplatform PRIVATE hpcplatform_WITH_SIMD_GENERIC=1)
  # Capability probes only: operator folders gate their own -mavx2 object
  # libraries on hpcplatform_COMPILER_HAS_AVX2 and define hpcplatform_ARCH_AVX2 themselves.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" hpcplatform_COMPILER_HAS_AVX2)
  endif()
endif()

# --- Optional backends ----------------------------------------------------
if(hpcplatform_WITH_OPENCV)
  find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc)
  target_sources(hpcplatform PRIVATE
    src/backends/opencv/opencv_backend.cpp
    src/backends/opencv/opencv_convert.cpp
    src/backends/opencv/opencv_memory_manager.cpp
    src/backends/opencv/opencv_tensor_access.cpp)
  target_include_directories(hpcplatform SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(hpcplatform PUBLIC ${OpenCV_LIBS})
  target_compile_definitions(hpcplatform PUBLIC hpcplatform_WITH_OPENCV=1)
endif()

if(hpcplatform_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(hpcplatform PRIVATE
    src/backends/cuda/cuda_backend.cpp
    src/backends/cuda/cuda_copy.cpp
    src/backends/cuda/cuda_copy_strided.cu
    src/backends/cuda/cuda_memory_manager.cpp
    src/backends/cuda/cuda_stream.cpp)
  target_link_libraries(hpcplatform PUBLIC CUDA::cudart)
  target_compile_definitions(hpcplatform PUBLIC hpcplatform_WITH_CUDA=1)
  target_compile_options(hpcplatform PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

# --- YAML config-driven operator loader (optional) ------------------------
# Pulls in yaml-cpp and the operator descriptors; the public LoadOperators API
# is always declared and returns kUnsupported when this is off (see
# src/ops/loader/).
if(hpcplatform_WITH_YAML)
  find_package(yaml-cpp REQUIRED)
  target_sources(hpcplatform PRIVATE
    src/ops/loader/load_operators_real.cpp
    src/ops/loader/register_descriptors.cpp
    src/ops/loader/yaml_parser.cpp)
  # PUBLIC so consumers (tests, samples, anything linking the static hpcplatform and
  # calling LoadOperators) get yaml-cpp's include dirs and link line, matching
  # how the OpenCV backend is linked. Public headers do not change based on
  # hpcplatform_WITH_YAML, so propagating it is ABI-neutral; it lets downstream code
  # #ifdef on the feature (e.g. the operator tests).
  target_link_libraries(hpcplatform PUBLIC yaml-cpp::yaml-cpp)
  target_compile_definitions(hpcplatform PUBLIC hpcplatform_WITH_YAML=1)
endif()

# Operator folders are self-contained (sources, optional-backend files,
# per-op tests/benchmarks); each one is mounted with a single line.
if(hpcplatform_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  # One GoogleTest executable per test translation unit, registered with CTest.
  # gtest_main supplies main(), so test files only define TEST(...) cases.
  function(hpcplatform_add_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE hpcplatform GTest::gtest_main)
    target_include_directories(${target} PRIVATE
      ${PROJECT_SOURCE_DIR}/tests
      ${PROJECT_SOURCE_DIR}/src)
    add_test(NAME ${target} COMMAND ${target})
    if(hpcplatform_ENABLE_SANITIZERS AND hpcplatform_WITH_CUDA)
      # The NVIDIA driver keeps process-global state alive after
      # cuDriverGetVersion(), which LeakSanitizer reports as a leak. ASan/UBSan
      # error detection stays enabled; only leak reporting is turned off.
      set_tests_properties(${target} PROPERTIES
        ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
    endif()
  endfunction()
endif()
add_subdirectory(src/ops)

include(GNUInstallDirs)
install(TARGETS hpcplatform EXPORT hpcplatformTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT hpcplatformTargets
  FILE hpcplatformTargets.cmake
  NAMESPACE hpcplatform::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hpcplatform)
install(FILES Readme.md DESTINATION ${CMAKE_INSTALL_DOCDIR})

if(hpcplatform_BUILD_TESTS)
  add_subdirectory(tests)
endif()

if(hpcplatform_BUILD_SAMPLES)
  add_subdirectory(samples)
endif()

# --- Packaging ------------------------------------------------------------
# Binary packages: cpack -G TGZ / cpack -G DEB. Source package: cpack --config
# CPackSourceConfig.cmake (or cpack -G TGZ --config ...). Package layout is
# exactly what the install() rules above describe.
set(CPACK_PACKAGE_NAME "hpcplatform")
set(CPACK_PACKAGE_VENDOR "hpcplatform")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_CONTACT "hpcplatform maintainers")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "hpcplatform")
set(CPACK_PACKAGE_CHECKSUM "SHA256")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/Readme.md")

set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_SOURCE_IGNORE_FILES
  "/build[^/]*/"
  "/\\.git/"
  "/\\.cache/")

if(UNIX AND NOT APPLE)
  set(CPACK_GENERATOR "TGZ;DEB")
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "hpcplatform maintainers")
  set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
elseif(APPLE)
  set(CPACK_GENERATOR "TGZ")
else()
  set(CPACK_GENERATOR "ZIP")
endif()

include(CPack)
