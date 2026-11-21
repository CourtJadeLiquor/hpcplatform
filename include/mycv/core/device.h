#ifndef MYCV_CORE_DEVICE_H_
#define MYCV_CORE_DEVICE_H_

#include <cstdint>
#include <string>

namespace mycv {

// 后端类型：实现提供者。选择必须显式，不做运行时探测或回退。
enum class BackendType : uint8_t {
  kNativeCpu = 0,  // 自研 CPU 内核
  kOpenCV = 1,     // OpenCV 实现
  kCUDA = 2,       // CUDA 实现
  kNPU = 3,        // NPU 占位
};

// 设备类型：硬件执行域。
enum class DeviceType : uint8_t {
  kCPU = 0,
  kGPU = 1,
  kNPU = 2,
};

// 返回后端的稳定名称。
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

// 返回设备类型的稳定名称。
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

// 返回后端默认使用的设备类型。
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

// 设备标识：类型 + 同类型内的序号（例如 cuda:0）。
struct Device {
  DeviceType type = DeviceType::kCPU;
  int index = 0;

  // 类型与序号都相同才算同一设备。
  bool operator==(const Device& other) const {
    return type == other.type && index == other.index;
  }
  bool operator!=(const Device& other) const { return !(*this == other); }

  // 格式化为 "type:index"。
  std::string ToString() const {
    return std::string(DeviceTypeName(type)) + ":" + std::to_string(index);
  }
};

}  // namespace mycv

#endif  // MYCV_CORE_DEVICE_H_
