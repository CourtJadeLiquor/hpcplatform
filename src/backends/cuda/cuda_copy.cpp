#include <cuda_runtime_api.h>

#include "backends/backend_internal.h"
#include "backends/cuda/cuda_backend.h"
#include "backends/cuda/cuda_common.h"
#include "backends/cuda/cuda_stream.h"

namespace hpcplatform
{

    Status CudaBackend::CopyBuffer(Buffer* dst,
                                   size_t dst_offset,
                                   const Buffer* src,
                                   size_t src_offset,
                                   size_t bytes,
                                   Stream* stream) const
    {
        if (dst == nullptr || src == nullptr || bytes == 0)
        {
            return Status::InvalidArgument("CUDA copy requires buffers and non-zero bytes");
        }
        if (dst_offset > dst->size() || bytes > dst->size() - dst_offset ||
            src_offset > src->size() || bytes > src->size() - src_offset)
        {
            return Status::InvalidArgument("CUDA copy range out of bounds");
        }
        if (src->kind() == MemoryKind::kDevice && dst->kind() == MemoryKind::kDevice &&
            src->device().index != dst->device().index)
        {
            return Status::Unsupported("peer-to-peer CUDA copies are not implemented");
        }

        const int device_index =
            src->kind() == MemoryKind::kDevice ? src->device().index : dst->device().index;
        hpcplatform_RETURN_IF_ERROR(CudaStatus(cudaSetDevice(device_index), "cudaSetDevice"));
        cudaMemcpyKind kind = cudaMemcpyDefault;
        if (src->kind() == MemoryKind::kDevice && dst->kind() == MemoryKind::kDevice)
        {
            kind = cudaMemcpyDeviceToDevice;
        }
        else if (src->kind() == MemoryKind::kDevice)
        {
            kind = cudaMemcpyDeviceToHost;
        }
        else if (dst->kind() == MemoryKind::kDevice)
        {
            kind = cudaMemcpyHostToDevice;
        }
        else
        {
            return Status::InvalidArgument("CudaCopyBuffer requires at least one device buffer");
        }

        CudaStream* cuda_stream = dynamic_cast<CudaStream*>(stream);
        cudaStream_t handle = cuda_stream != nullptr ? cuda_stream->handle() : nullptr;
        if (cuda_stream == nullptr)
        {
            // D2H uses the destination CPU Context's stream. Serialize with pending
            // CUDA work before using the default copy stream rather than racing a
            // non-blocking stream.
            hpcplatform_RETURN_IF_ERROR(
                CudaStatus(cudaDeviceSynchronize(), "cudaDeviceSynchronize"));
        }
        hpcplatform_RETURN_IF_ERROR(
            CudaStatus(cudaMemcpyAsync(static_cast<char*>(dst->data()) + dst_offset,
                                       static_cast<const char*>(src->data()) + src_offset, bytes,
                                       kind, handle),
                       "cudaMemcpyAsync"));
        return CudaStatus(cudaStreamSynchronize(handle), "cudaStreamSynchronize(copy)");
    }

    Status CudaBackend::CopyStrided(Buffer* dst,
                                    size_t dst_offset,
                                    const int64_t* dst_strides,
                                    const Buffer* src,
                                    size_t src_offset,
                                    const int64_t* src_strides,
                                    const Shape& shape,
                                    size_t element_size,
                                    Stream* stream) const
    {
        if (dst->kind() != MemoryKind::kDevice && src->kind() != MemoryKind::kDevice)
        {
            // Host-to-host under a CUDA-bound executor: the portable loop suffices.
            return Backend::CopyStrided(dst, dst_offset, dst_strides, src, src_offset, src_strides,
                                        shape, element_size, stream);
        }
        return CudaCopyStrided(dst, dst_offset, dst_strides, src, src_offset, src_strides, shape,
                               element_size, stream);
    }

}  // namespace hpcplatform
