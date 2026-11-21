#ifndef MYCV_SRC_BACKENDS_BACKEND_INTERNAL_H_
#define MYCV_SRC_BACKENDS_BACKEND_INTERNAL_H_

#include <memory>

#include "core/backend.h"

namespace mycv {

// Backend-specific creators implemented in each backend folder. The public
// entry point CreateBackend (declared in core/backend.h) dispatches to them.
Result<std::shared_ptr<Backend>> CreateCpuBackend(int device_index);
Result<std::shared_ptr<Backend>> CreateOpenCvBackend(int device_index);
Result<std::shared_ptr<Backend>> CreateCudaBackend(int device_index);
Result<std::shared_ptr<Backend>> CreateNpuBackend(int device_index);

// CUDA device-to-device transfer helper used by CudaBackend::CopyBuffer.
Status CudaCopyBuffer(Buffer* dst, size_t dst_offset, const Buffer* src,
                      size_t src_offset, size_t bytes, Stream* stream);

}  // namespace mycv

#endif  // MYCV_SRC_BACKENDS_BACKEND_INTERNAL_H_
