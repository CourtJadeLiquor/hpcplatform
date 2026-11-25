#ifndef hpcplatform_SRC_BACKENDS_BACKEND_INTERNAL_H_
#define hpcplatform_SRC_BACKENDS_BACKEND_INTERNAL_H_

#include <memory>

#include "core/backend.h"

namespace hpcplatform {

// Backend-specific creators implemented in each backend folder. The public
// entry point CreateBackend (declared in core/backend.h) dispatches to them.
Result<std::shared_ptr<Backend>> CreateCpuBackend(int device_index);
Result<std::shared_ptr<Backend>> CreateOpenCvBackend(int device_index);
Result<std::shared_ptr<Backend>> CreateCudaBackend(int device_index);
Result<std::shared_ptr<Backend>> CreateNpuBackend(int device_index);

// CUDA device-to-device transfer helper used by CudaBackend::CopyBuffer.
Status CudaCopyBuffer(Buffer* dst, size_t dst_offset, const Buffer* src,
                      size_t src_offset, size_t bytes, Stream* stream);

// CUDA strided-copy helper used by CudaBackend::CopyStrided. Launches a single
// grid-stride kernel that materializes any strided view pair; requires at
// least one device operand and rejects peer-to-peer copies across devices.
// Defined in cuda_copy_strided.cu.
Status CudaCopyStrided(Buffer* dst, size_t dst_offset,
                       const int64_t* dst_strides, const Buffer* src,
                       size_t src_offset, const int64_t* src_strides,
                       const Shape& shape, size_t element_size,
                       Stream* stream);

}  // namespace hpcplatform

#endif  // hpcplatform_SRC_BACKENDS_BACKEND_INTERNAL_H_
