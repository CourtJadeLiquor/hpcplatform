#ifndef MYCV_SRC_BACKENDS_BACKENDS_INTERNAL_H_
#define MYCV_SRC_BACKENDS_BACKENDS_INTERNAL_H_

#include <memory>

#include "mycv/backends/backend.h"

namespace mycv {

class Tensor;

Result<std::shared_ptr<Backend>> CreateCpuBackend(int device_index);
Result<std::shared_ptr<Backend>> CreateOpenCvBackend(int device_index);
Result<std::shared_ptr<Backend>> CreateCudaBackend(int device_index);
Result<std::shared_ptr<Backend>> CreateNpuBackend(int device_index);

Status CudaCopyBuffer(Buffer* dst, size_t dst_offset, const Buffer* src,
                      size_t src_offset, size_t bytes, Stream* stream);

// Element-wise copy used by Context::Copy and Tensor::Clone. The backend that
// owns the device side drives the transfer; host-only copies use `backend`.
Status CopyTensor(Backend& backend, Tensor* dst, const Tensor& src,
                  Stream* stream);

// Returns the backend that allocated the tensor. Buffers outlive their Context,
// so when the original backend is gone a fresh one is created for the transfer
// instead of dereferencing a dead allocator.
Result<std::shared_ptr<Backend>> ResolveOwningBackend(const Tensor& tensor);

}  // namespace mycv

#endif  // MYCV_SRC_BACKENDS_BACKENDS_INTERNAL_H_
