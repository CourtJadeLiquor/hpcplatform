#ifndef MYCV_SRC_BACKENDS_CUDA_CUDA_STREAM_H_
#define MYCV_SRC_BACKENDS_CUDA_CUDA_STREAM_H_

#include <cuda_runtime_api.h>

#include "mycv/core/stream.h"

namespace mycv {

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

}  // namespace mycv

#endif  // MYCV_SRC_BACKENDS_CUDA_CUDA_STREAM_H_
