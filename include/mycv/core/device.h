#ifndef MYCV_CORE_DEVICE_H_
#define MYCV_CORE_DEVICE_H_

#include <cstdint>
#include <string>

#include "mycv/core/error.h"

namespace mycv {

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

}  // namespace mycv

#endif  // MYCV_CORE_DEVICE_H_
