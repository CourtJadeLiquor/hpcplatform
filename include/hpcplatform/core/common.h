#ifndef hpcplatform_CORE_COMMON_H_
#define hpcplatform_CORE_COMMON_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include "hpcplatform/core/error.h"

namespace hpcplatform {

// ===========================================================================
// Element types
// ===========================================================================

// Element types supported by the core. Backends may support only a subset;
// availability is queried at op level.
enum class DataType : uint8_t {
  kUInt8 = 0,
  kInt8 = 1,
  kUInt16 = 2,
  kInt16 = 3,
  kFloat16 = 4,
  kFloat32 = 5,
  kFloat64 = 6,
  kInt32 = 7,
  kInt64 = 8,
};

// Returns the element size in bytes; returns 0 for unknown types.
inline size_t DTypeSize(DataType dtype) {
  switch (dtype) {
    case DataType::kUInt8:
    case DataType::kInt8:
      return 1;
    case DataType::kUInt16:
    case DataType::kInt16:
    case DataType::kFloat16:
      return 2;
    case DataType::kFloat32:
    case DataType::kInt32:
      return 4;
    case DataType::kFloat64:
    case DataType::kInt64:
      return 8;
  }
  return 0;
}

// Returns a stable type name for logs and error messages.
inline const char* DTypeName(DataType dtype) {
  switch (dtype) {
    case DataType::kUInt8:
      return "uint8";
    case DataType::kInt8:
      return "int8";
    case DataType::kUInt16:
      return "uint16";
    case DataType::kInt16:
      return "int16";
    case DataType::kFloat16:
      return "float16";
    case DataType::kFloat32:
      return "float32";
    case DataType::kFloat64:
      return "float64";
    case DataType::kInt32:
      return "int32";
    case DataType::kInt64:
      return "int64";
  }
  return "unknown";
}

// Returns true for floating-point types (f16/f32/f64).
inline bool IsFloatingPoint(DataType dtype) {
  return dtype == DataType::kFloat16 || dtype == DataType::kFloat32 ||
         dtype == DataType::kFloat64;
}

// ===========================================================================
// Backends and devices
// ===========================================================================

// Backend type: implementation provider. Selection is explicit, with no runtime
// detection or fallback.
enum class BackendType : uint8_t {
  kNativeCpu = 0,  // Native CPU kernels
  kOpenCV = 1,     // OpenCV implementation
  kCUDA = 2,       // CUDA implementation
  kNPU = 3,        // NPU placeholder
};

// Device type: hardware execution domain.
enum class DeviceType : uint8_t {
  kCPU = 0,
  kGPU = 1,
  kNPU = 2,
};

// Returns the backend's stable name.
inline const char* BackendTypeName(BackendType type) {
  switch (type) {
    case BackendType::kNativeCpu:
      return "native_cpu";
    case BackendType::kOpenCV:
      return "opencv";
    case BackendType::kCUDA:
      return "cuda";
    case BackendType::kNPU:
      return "npu";
  }
  return "unknown";
}

// Parses a stable backend name back to its enum. Returns kInvalidArgument for
// an unrecognized name; the inverse of BackendTypeName (excluding "unknown").
inline Result<BackendType> BackendTypeFromName(const std::string& name) {
  if (name == "native_cpu") return BackendType::kNativeCpu;
  if (name == "opencv") return BackendType::kOpenCV;
  if (name == "cuda") return BackendType::kCUDA;
  if (name == "npu") return BackendType::kNPU;
  return Status::InvalidArgument("unknown backend name: " + name);
}

// Returns the device type's stable name.
inline const char* DeviceTypeName(DeviceType type) {
  switch (type) {
    case DeviceType::kCPU:
      return "cpu";
    case DeviceType::kGPU:
      return "gpu";
    case DeviceType::kNPU:
      return "npu";
  }
  return "unknown";
}

// Returns the default device type for a backend.
inline DeviceType DefaultDeviceType(BackendType backend) {
  switch (backend) {
    case BackendType::kNativeCpu:
    case BackendType::kOpenCV:
      return DeviceType::kCPU;
    case BackendType::kCUDA:
      return DeviceType::kGPU;
    case BackendType::kNPU:
      return DeviceType::kNPU;
  }
  return DeviceType::kCPU;
}

// Device identifier: type + index within the same type (e.g., cuda:0).
struct Device {
  DeviceType type = DeviceType::kCPU;
  int index = 0;

  // Devices are equal only when both type and index match.
  bool operator==(const Device& other) const {
    return type == other.type && index == other.index;
  }
  bool operator!=(const Device& other) const { return !(*this == other); }

  // Formats the device as "type:index".
  std::string ToString() const {
    return std::string(DeviceTypeName(type)) + ":" + std::to_string(index);
  }
};

// ===========================================================================
// CPU kernel levels
// ===========================================================================

// Selects which CPU implementation of an op runs. This is an explicit,
// compile-time-enabled knob (no auto-tuning, no CPU feature probing).
// Non-CPU backends accept kNaive only and use their own best implementation.
enum class CpuKernelLevel : uint8_t {
  kNaive = 0,        // portable reference implementation, used as ground truth
  kOpenMP = 1,       // thread-parallel reference implementation
  kSimdGeneric = 2,  // portable SIMD (compiler vectorization / intrinsics)
  kSimdAvx2 = 3,     // x86-64 AVX2
  kSimdNeon = 4,     // aarch64 NEON
  kTbb = 5,          // oneTBB thread-parallel implementation
};

inline const char* CpuKernelLevelName(CpuKernelLevel level) {
  switch (level) {
    case CpuKernelLevel::kNaive:
      return "naive";
    case CpuKernelLevel::kOpenMP:
      return "openmp";
    case CpuKernelLevel::kSimdGeneric:
      return "simd_generic";
    case CpuKernelLevel::kSimdAvx2:
      return "simd_avx2";
    case CpuKernelLevel::kSimdNeon:
      return "simd_neon";
    case CpuKernelLevel::kTbb:
      return "tbb";
  }
  return "unknown";
}

// Parses a stable CPU kernel level name back to its enum. Inverse of
// CpuKernelLevelName (excluding "unknown"); unknown name -> kInvalidArgument.
inline Result<CpuKernelLevel> CpuKernelLevelFromName(const std::string& name) {
  if (name == "naive") return CpuKernelLevel::kNaive;
  if (name == "openmp") return CpuKernelLevel::kOpenMP;
  if (name == "simd_generic") return CpuKernelLevel::kSimdGeneric;
  if (name == "simd_avx2") return CpuKernelLevel::kSimdAvx2;
  if (name == "simd_neon") return CpuKernelLevel::kSimdNeon;
  if (name == "tbb") return CpuKernelLevel::kTbb;
  return Status::InvalidArgument("unknown cpu kernel level name: " + name);
}

// Returns true for SIMD levels (generic / AVX2 / NEON).
inline bool IsSimdKernelLevel(CpuKernelLevel level) {
  return level == CpuKernelLevel::kSimdGeneric ||
         level == CpuKernelLevel::kSimdAvx2 ||
         level == CpuKernelLevel::kSimdNeon;
}

// ===========================================================================
// Memory category
// ===========================================================================

// Memory category; determines the allocator and copy path.
enum class MemoryKind : uint8_t {
  kHost = 0,        // Standard pageable host memory
  kHostPinned = 1,  // Pinned host memory (suitable for DMA)
  kDevice = 2,      // Device memory (GPU / NPU)
};

// ===========================================================================
// Shape
// ===========================================================================

// Fixed-capacity shape descriptor (no heap allocation in hot paths).
// Precondition: no more than kMaxDims dimensions are passed to the constructor.
class Shape {
 public:
  static constexpr int kMaxDims = 8;

  Shape() = default;

  // Keeps at most kMaxDims dimensions and truncates the rest; callers should
  // avoid longer lists.
  Shape(std::initializer_list<int64_t> dims) {
    rank_ = static_cast<int>(dims.size() < static_cast<size_t>(kMaxDims)
                                 ? dims.size()
                                 : static_cast<size_t>(kMaxDims));
    int i = 0;
    for (int64_t dim : dims) {
      if (i >= rank_) break;
      dims_[static_cast<size_t>(i++)] = dim;
    }
  }

  // Constructs from a vector, likewise keeping at most kMaxDims dimensions.
  explicit Shape(const std::vector<int64_t>& dims) {
    rank_ = static_cast<int>(dims.size() < static_cast<size_t>(kMaxDims)
                                 ? dims.size()
                                 : static_cast<size_t>(kMaxDims));
    for (int i = 0; i < rank_; ++i) {
      dims_[static_cast<size_t>(i)] = dims[static_cast<size_t>(i)];
    }
  }

  // Returns the number of dimensions.
  int Rank() const { return rank_; }
  // Returns the length of dimension index; caller must ensure
  // 0 <= index < Rank().
  int64_t Dim(int index) const { return dims_[static_cast<size_t>(index)]; }
  // Returns the contiguous dimension array, whose length is Rank().
  const int64_t* data() const { return dims_.data(); }

  // Scalar (rank 0) has exactly one element.
  int64_t Numel() const {
    int64_t numel = 1;
    for (int i = 0; i < rank_; ++i)
      numel *= dims_[static_cast<size_t>(i)];
    return numel;
  }

  // Returns true for scalars (rank 0).
  bool IsScalar() const { return rank_ == 0; }

  // Semantic validation (dims must be positive). Construction itself is
  // unchecked so Shape stays trivially cheap inside kernels.
  Status Validate() const {
    for (int i = 0; i < rank_; ++i) {
      if (dims_[static_cast<size_t>(i)] <= 0) {
        return Status::InvalidArgument("shape dims must be positive: " +
                                       ToString());
      }
    }
    return Status::Ok();
  }

  // Compares rank and every dimension length.
  bool operator==(const Shape& other) const {
    if (rank_ != other.rank_) return false;
    for (int i = 0; i < rank_; ++i) {
      if (dims_[static_cast<size_t>(i)] !=
          other.dims_[static_cast<size_t>(i)]) {
        return false;
      }
    }
    return true;
  }
  bool operator!=(const Shape& other) const { return !(*this == other); }

  // Formats as "[d0,d1,...]" for error messages and logs.
  std::string ToString() const;

 private:
  std::array<int64_t, kMaxDims> dims_{};
  int rank_ = 0;
};

}  // namespace hpcplatform

#endif  // hpcplatform_CORE_COMMON_H_
