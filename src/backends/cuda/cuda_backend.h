#ifndef MYCV_SRC_BACKENDS_CUDA_CUDA_BACKEND_H_
#define MYCV_SRC_BACKENDS_CUDA_CUDA_BACKEND_H_

#include <string>

#include "backends/cuda/cuda_memory_manager.h"
#include "backends/cuda/cuda_stream.h"
#include "core/backend.h"

namespace mycv {

class CudaBackend final : public Backend {
 public:
  explicit CudaBackend(int device_index)
      : device_{DeviceType::kGPU, device_index},
        memory_(device_),
        stream_(device_) {}

  Status Initialize();
  BackendType type() const override { return BackendType::kCUDA; }
  Device device() const override { return device_; }
  MemoryManager& memory() override { return memory_; }
  // Grants mutable access to the owned stream; the backend itself is not
  // modified, so the accessor is const and casts away the member's constness.
  Stream& stream() const override { return const_cast<CudaStream&>(stream_); }
  Status Synchronize() override { return stream_.Synchronize(); }
  Status CopyBuffer(Buffer* dst, size_t dst_offset, const Buffer* src,
                    size_t src_offset, size_t bytes,
                    Stream* stream) const override;
  bool SupportsKernelLevel(CpuKernelLevel level) const override {
    return level == CpuKernelLevel::kNaive;
  }
  std::string Describe() const override;

 private:
  Device device_;
  CudaMemoryManager memory_;
  CudaStream stream_;
  std::string device_name_;
  int compute_major_ = 0;
  int compute_minor_ = 0;
};

}  // namespace mycv

#endif  // MYCV_SRC_BACKENDS_CUDA_CUDA_BACKEND_H_
