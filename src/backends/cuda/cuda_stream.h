#ifndef hpcplatform_SRC_BACKENDS_CUDA_CUDA_STREAM_H_
#define hpcplatform_SRC_BACKENDS_CUDA_CUDA_STREAM_H_

#include <cuda_runtime_api.h>

#include "hpcplatform/core/stream.h"

namespace hpcplatform {

class CudaStream final : public Stream {
 public:
  explicit CudaStream(Device device) : device_(device) {}
  ~CudaStream() override;

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  Status Initialize();
  Device device() const override { return device_; }
  Status Synchronize() override;
  bool synchronous() const override { return false; }
  cudaStream_t handle() const { return stream_; }

 private:
  Device device_;
  cudaStream_t stream_ = nullptr;
};

}  // namespace hpcplatform

#endif  // hpcplatform_SRC_BACKENDS_CUDA_CUDA_STREAM_H_
