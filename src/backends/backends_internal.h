#ifndef MYCV_SRC_BACKENDS_BACKENDS_INTERNAL_H_
#define MYCV_SRC_BACKENDS_BACKENDS_INTERNAL_H_

#include <memory>

#include "mycv/backends/backend.h"

namespace mycv {

Result<std::shared_ptr<Backend>> CreateCpuBackend(int device_index);
Result<std::shared_ptr<Backend>> CreateOpenCvBackend(int device_index);
Result<std::shared_ptr<Backend>> CreateCudaBackend(int device_index);
Result<std::shared_ptr<Backend>> CreateNpuBackend(int device_index);

Status CudaCopyBuffer(Buffer* dst, size_t dst_offset, const Buffer* src,
                      size_t src_offset, size_t bytes, Stream* stream);

}  // namespace mycv

#endif  // MYCV_SRC_BACKENDS_BACKENDS_INTERNAL_H_
