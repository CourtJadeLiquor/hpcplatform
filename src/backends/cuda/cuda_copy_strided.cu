// Strided device copy: a single grid-stride kernel materializes any strided
// view pair, so a non-contiguous copy costs one launch instead of one
// synchronous transfer per element. This is CudaBackend's CopyStrided
// strategy — stride handling is a backend decision, core only routes.

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

#include "backends/backend_internal.h"
#include "backends/cuda/cuda_common.h"
#include "backends/cuda/cuda_stream.h"
#include "hpcplatform/core/common.h"
#include "hpcplatform/core/error.h"
#include "hpcplatform/core/memory.h"

namespace hpcplatform {
namespace {

// Kernel parameters are value-copied into constant memory, so no auxiliary
// device allocation is needed for the layout description. The arrays live in
// a struct because a bare array parameter would decay to a pointer and hand
// the kernel a host address.
constexpr int kMaxRank = 8;

struct StridedLayout {
  long long extents[kMaxRank];
  long long dst_strides[kMaxRank];
  long long src_strides[kMaxRank];
};

__global__ void StridedCopyKernel(char* dst, const char* src, int rank,
                                  StridedLayout layout, size_t element_size,
                                  long long numel) {
  long long index = static_cast<long long>(blockIdx.x) * blockDim.x +
                    static_cast<long long>(threadIdx.x);
  const long long stride =
      static_cast<long long>(gridDim.x) * static_cast<long long>(blockDim.x);
  for (; index < numel; index += stride) {
    long long remaining = index;
    size_t dst_byte = 0;
    size_t src_byte = 0;
    for (int dim = rank - 1; dim >= 0; --dim) {
      const long long coord = remaining % layout.extents[dim];
      remaining /= layout.extents[dim];
      dst_byte += static_cast<size_t>(coord) *
                  static_cast<size_t>(layout.dst_strides[dim]);
      src_byte += static_cast<size_t>(coord) *
                  static_cast<size_t>(layout.src_strides[dim]);
    }
    for (size_t byte = 0; byte < element_size; ++byte) {
      dst[dst_byte + byte] = src[src_byte + byte];
    }
  }
}

// Returns whether the strides describe the standard row-major packed layout.
bool IsPacked(const Shape& shape, const int64_t* strides,
              size_t element_size) {
  int64_t expected = 1;
  for (int dim = shape.Rank() - 1; dim >= 0; --dim) {
    if (strides[dim] != expected * static_cast<int64_t>(element_size)) {
      return false;
    }
    expected *= shape.Dim(dim);
  }
  return true;
}

}  // namespace

Status CudaCopyStrided(Buffer* dst, size_t dst_offset,
                       const int64_t* dst_strides, const Buffer* src,
                       size_t src_offset, const int64_t* src_strides,
                       const Shape& shape, size_t element_size,
                       Stream* stream) {
  if (dst == nullptr || src == nullptr || element_size == 0) {
    return Status::InvalidArgument(
        "CUDA strided copy requires buffers and non-zero elements");
  }
  const int rank = shape.Rank();
  if (rank <= 0 || rank > kMaxRank) {
    return Status::InvalidArgument("CUDA strided copy supports rank 1.." +
                                   std::to_string(kMaxRank));
  }
  if (src->kind() == MemoryKind::kDevice &&
      dst->kind() == MemoryKind::kDevice &&
      src->device().index != dst->device().index) {
    return Status::Unsupported("peer-to-peer CUDA copies are not implemented");
  }

  const int device_index = src->kind() == MemoryKind::kDevice
                               ? src->device().index
                               : dst->device().index;
  hpcplatform_RETURN_IF_ERROR(
      CudaStatus(cudaSetDevice(device_index), "cudaSetDevice"));

  CudaStream* cuda_stream = dynamic_cast<CudaStream*>(stream);
  cudaStream_t handle = cuda_stream != nullptr ? cuda_stream->handle() : nullptr;
  if (cuda_stream == nullptr) {
    // Serialize with pending CUDA work before using the default copy stream
    // rather than racing a non-blocking stream.
    hpcplatform_RETURN_IF_ERROR(
        CudaStatus(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));
  }

  if (IsPacked(shape, dst_strides, element_size) &&
      IsPacked(shape, src_strides, element_size)) {
    // Packed views collapse to a single contiguous block transfer.
    const size_t bytes = static_cast<size_t>(shape.Numel()) * element_size;
    hpcplatform_RETURN_IF_ERROR(CudaStatus(
        cudaMemcpyAsync(static_cast<char*>(dst->data()) + dst_offset,
                        static_cast<const char*>(src->data()) + src_offset,
                        bytes, cudaMemcpyDefault, handle),
        "cudaMemcpyAsync"));
    return CudaStatus(cudaStreamSynchronize(handle),
                      "cudaStreamSynchronize(copy)");
  }

  StridedLayout layout = {};
  for (int dim = 0; dim < rank; ++dim) {
    layout.extents[dim] = shape.Dim(dim);
    layout.dst_strides[dim] = dst_strides[dim];
    layout.src_strides[dim] = src_strides[dim];
  }
  const long long numel = shape.Numel();
  const int threads = 256;
  const int blocks = static_cast<int>(
      (numel + threads - 1) / threads < 65535
          ? (numel + threads - 1) / threads
          : 65535);
  StridedCopyKernel<<<blocks, threads, 0, handle>>>(
      static_cast<char*>(dst->data()) + dst_offset,
      static_cast<const char*>(src->data()) + src_offset, rank, layout,
      element_size, numel);
  hpcplatform_RETURN_IF_ERROR(
      CudaStatus(cudaGetLastError(), "StridedCopyKernel launch"));
  return CudaStatus(cudaStreamSynchronize(handle),
                    "cudaStreamSynchronize(copy)");
}

}  // namespace hpcplatform
