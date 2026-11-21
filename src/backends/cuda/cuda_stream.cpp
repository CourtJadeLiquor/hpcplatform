#include "backends/cuda/cuda_stream.h"

#include "backends/cuda/cuda_common.h"

namespace mycv {

CudaStream::~CudaStream() {
  if (stream_ != nullptr) {
    cudaSetDevice(device_.index);
    cudaStreamDestroy(stream_);
  }
}

Status CudaStream::Initialize() {
  if (stream_ != nullptr)
    return Status::BadState("CUDA stream is already initialized");
  MYCV_RETURN_IF_ERROR(
      CudaStatus(cudaSetDevice(device_.index), "cudaSetDevice"));
  return CudaStatus(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                    "cudaStreamCreateWithFlags");
}

Status CudaStream::Synchronize() {
  if (stream_ == nullptr)
    return Status::BadState("CUDA stream is not initialized");
  MYCV_RETURN_IF_ERROR(
      CudaStatus(cudaSetDevice(device_.index), "cudaSetDevice"));
  return CudaStatus(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
}

}  // namespace mycv
