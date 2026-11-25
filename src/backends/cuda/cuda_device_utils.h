#ifndef hpcplatform_SRC_BACKENDS_CUDA_CUDA_DEVICE_UTILS_H_
#define hpcplatform_SRC_BACKENDS_CUDA_CUDA_DEVICE_UTILS_H_

// Backend service header for operator CUDA implementations
// (docs/architecture.md §6.2): device-side tensor/image descriptors, thread
// indexing, and launch helpers shared by every *_cuda.cu. Operators include
// this header; it is never exposed through the public API.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "backends/cuda/cuda_common.h"
#include "backends/cuda/cuda_stream.h"
#include "framework/image_utils.h"
#include "hpcplatform/core/tensor.h"
#include "hpcplatform/ops/image.h"
#include "hpcplatform/ops/op.h"

namespace hpcplatform
{
    namespace ops_internal
    {

        constexpr unsigned int kThreadsPerBlock = 256;
        constexpr int64_t kMaxBlocks = 65535;

        struct DeviceTensor
        {
            const void* data;
            int rank;
            int64_t dims[Shape::kMaxDims];
            int64_t strides[Shape::kMaxDims];
        };

        struct DeviceImage
        {
            int rank;
            int64_t n;
            int64_t h;
            int64_t w;
            int64_t c;
            ImageLayout layout;
        };

        struct DeviceAffine
        {
            double values[6];
        };

        inline DeviceTensor Describe(const Tensor& tensor)
        {
            DeviceTensor descriptor = {};
            descriptor.data = tensor.data();
            descriptor.rank = tensor.shape().Rank();
            for (int dim = 0; dim < descriptor.rank; ++dim)
            {
                descriptor.dims[dim] = tensor.shape().Dim(dim);
                descriptor.strides[dim] = tensor.Stride(dim);
            }
            return descriptor;
        }

        inline DeviceImage Describe(const ImageInfo& image)
        {
            DeviceImage descriptor = {};
            descriptor.rank = image.rank;
            descriptor.n = image.n;
            descriptor.h = image.h;
            descriptor.w = image.w;
            descriptor.c = image.c;
            descriptor.layout = image.layout;
            return descriptor;
        }

        inline unsigned int BlockCount(int64_t count)
        {
            const int64_t needed = 1 + (count - 1) / kThreadsPerBlock;
            return static_cast<unsigned int>(needed < kMaxBlocks ? needed : kMaxBlocks);
        }

        inline Result<CudaStream*> GetCudaStream(const OpContext& context)
        {
            Stream* stream = context.stream();
            if (stream == nullptr && context.context() != nullptr)
            {
                stream = &context.context()->stream();
            }
            if (stream == nullptr)
            {
                return Status::InvalidArgument(
                    "CUDA kernel requires OpContext.stream() or OpContext.context()");
            }
            CudaStream* cuda_stream = dynamic_cast<CudaStream*>(stream);
            if (cuda_stream == nullptr)
            {
                return Status::InvalidArgument("CUDA kernel requires a CudaStream");
            }
            if (cuda_stream->handle() == nullptr)
            {
                return Status::BadState("CUDA stream is not initialized");
            }
            return cuda_stream;
        }

        inline Status FinishLaunch(cudaStream_t stream, const char* operation)
        {
            const Status launch_status = CudaStatus(cudaGetLastError(), operation);
            const Status synchronize_status =
                CudaStatus(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
            if (!launch_status.ok()) return launch_status;
            return synchronize_status;
        }

        __device__ inline int64_t ThreadIndex()
        {
            return static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        }

        __device__ inline int64_t ThreadStride()
        {
            return static_cast<int64_t>(blockDim.x) * gridDim.x;
        }

        __device__ inline int64_t StridedElementOffset(const DeviceTensor& tensor,
                                                       int64_t linear_index)
        {
            int64_t offset = 0;
            for (int dim = tensor.rank - 1; dim >= 0; --dim)
            {
                const int64_t coordinate = linear_index % tensor.dims[dim];
                linear_index /= tensor.dims[dim];
                offset += coordinate * tensor.strides[dim];
            }
            return offset;
        }

        template <typename T>
        __device__ inline T SaturatingCast(double value)
        {
            static_assert(std::is_integral<T>::value, "integral destination required");
            if (::isnan(value)) return static_cast<T>(0);
            const double low = static_cast<double>(std::numeric_limits<T>::lowest());
            const double high = static_cast<double>(std::numeric_limits<T>::max());
            if (value <= low) return std::numeric_limits<T>::lowest();
            if (value >= high) return std::numeric_limits<T>::max();
            return static_cast<T>(value);
        }

        __device__ inline void DecodeImageIndex(const DeviceImage& image,
                                                int64_t linear_index,
                                                int64_t& n,
                                                int64_t& y,
                                                int64_t& x,
                                                int64_t& c)
        {
            if (image.layout == ImageLayout::kNHWC)
            {
                c = linear_index % image.c;
                linear_index /= image.c;
                x = linear_index % image.w;
                linear_index /= image.w;
                y = linear_index % image.h;
                n = linear_index / image.h;
            }
            else
            {
                x = linear_index % image.w;
                linear_index /= image.w;
                y = linear_index % image.h;
                linear_index /= image.h;
                c = linear_index % image.c;
                n = linear_index / image.c;
            }
        }

        __device__ inline int64_t ImageElementOffset(const DeviceTensor& tensor,
                                                     const DeviceImage& image,
                                                     int64_t n,
                                                     int64_t y,
                                                     int64_t x,
                                                     int64_t c)
        {
            int64_t coordinates[4] = {};
            if (image.rank == 2)
            {
                coordinates[0] = y;
                coordinates[1] = x;
            }
            else if (image.rank == 3 && image.layout == ImageLayout::kNHWC)
            {
                coordinates[0] = y;
                coordinates[1] = x;
                coordinates[2] = c;
            }
            else if (image.rank == 3)
            {
                coordinates[0] = c;
                coordinates[1] = y;
                coordinates[2] = x;
            }
            else if (image.layout == ImageLayout::kNHWC)
            {
                coordinates[0] = n;
                coordinates[1] = y;
                coordinates[2] = x;
                coordinates[3] = c;
            }
            else
            {
                coordinates[0] = n;
                coordinates[1] = c;
                coordinates[2] = y;
                coordinates[3] = x;
            }
            int64_t offset = 0;
            for (int dim = 0; dim < image.rank; ++dim)
            {
                offset += coordinates[dim] * tensor.strides[dim];
            }
            return offset;
        }

        template <typename T>
        __device__ inline void StoreImageValue(T* output, int64_t index, double value)
        {
            if constexpr (std::is_same<T, uint8_t>::value)
            {
                output[index] = SaturatingCast<uint8_t>(::round(value));
            }
            else
            {
                output[index] = static_cast<T>(value);
            }
        }

    }  // namespace ops_internal
}  // namespace hpcplatform

#endif  // hpcplatform_SRC_BACKENDS_CUDA_CUDA_DEVICE_UTILS_H_
