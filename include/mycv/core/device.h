#ifndef MYCV_CORE_DEVICE_H_
#define MYCV_CORE_DEVICE_H_

#include <cstdint>
#include <string>

namespace mycv {

// A backend is an implementation provider (native CPU kernels, OpenCV, CUDA,
// future NPU SDK). A device is the hardware it executes on. Backend selection
// is always explicit: there is no runtime auto-detection or silent fallback.
enum class BackendType : uint8_t {
  kNativeCpu = 0,
  kOpenCV = 1,
  kCUDA = 2,
  kNPU = 3,
};

enum class DeviceType : uint8_t {
  kCPU = 0,
  kGPU = 1,
  kNPU = 2,
};

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

struct Device {
  DeviceType type = DeviceType::kCPU;
  int index = 0;

  bool operator==(const Device& other) const {
    return type == other.type && index == other.index;
  }
  bool operator!=(const Device& other) const { return !(*this == other); }

  std::string ToString() const {
    return std::string(DeviceTypeName(type)) + ":" + std::to_string(index);
  }
};

}  // namespace mycv

#endif  // MYCV_CORE_DEVICE_H_
