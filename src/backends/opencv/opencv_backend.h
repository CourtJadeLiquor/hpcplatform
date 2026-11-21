#ifndef MYCV_SRC_BACKENDS_OPENCV_OPENCV_BACKEND_H_
#define MYCV_SRC_BACKENDS_OPENCV_OPENCV_BACKEND_H_

#include "backends/cpu/cpu_stream.h"
#include "backends/opencv/opencv_memory_manager.h"
#include "backends/service.h"

namespace mycv {

class OpenCvBackend final : public Backend {
 public:
  explicit OpenCvBackend(int device_index)
      : device_{DeviceType::kCPU, device_index}, memory_(device_), stream_(device_) {}

  BackendType type() const override { return BackendType::kOpenCV; }
  Device device() const override { return device_; }
  MemoryManager& memory() override { return memory_; }
  Stream& stream() override { return stream_; }
  Status Synchronize() override { return stream_.Synchronize(); }
  bool SupportsDType(DataType dtype) const override {
    return dtype == DataType::kUInt8 || dtype == DataType::kFloat32 ||
           dtype == DataType::kFloat64;
  }
  bool SupportsKernelLevel(CpuKernelLevel level) const override {
    return level == CpuKernelLevel::kNaive;
  }
  std::string Describe() const override {
    return "OpenCvBackend(device=" + device_.ToString() +
           ", dtypes=u8/f32/f64, layouts=HW/HWC/NHWC)";
  }

 private:
  Device device_;
  OpenCvMemoryManager memory_;
  CpuStream stream_;
};

}  // namespace mycv

#endif  // MYCV_SRC_BACKENDS_OPENCV_OPENCV_BACKEND_H_
