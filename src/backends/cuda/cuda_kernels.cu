#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "backends/cuda/cuda_common.h"
#include "backends/cuda/cuda_stream.h"
#include "ops/image_utils.h"
#include "ops/op_params.h"
#include "ops/op_registry.h"

namespace mycv {
namespace ops_internal {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
constexpr int64_t kMaxBlocks = 65535;

struct DeviceTensor {
  const void* data;
  int rank;
  int64_t dims[Shape::kMaxDims];
  int64_t strides[Shape::kMaxDims];
};

struct DeviceImage {
  int rank;
  int64_t n;
  int64_t h;
  int64_t w;
  int64_t c;
  ImageLayout layout;
};

struct DeviceAffine {
  double values[6];
};

DeviceTensor Describe(const Tensor& tensor) {
  DeviceTensor descriptor = {};
  descriptor.data = tensor.data();
  descriptor.rank = tensor.shape().Rank();
  for (int dim = 0; dim < descriptor.rank; ++dim) {
    descriptor.dims[dim] = tensor.shape().Dim(dim);
    descriptor.strides[dim] = tensor.Stride(dim);
  }
  return descriptor;
}

DeviceImage Describe(const ImageInfo& image) {
  DeviceImage descriptor = {};
  descriptor.rank = image.rank;
  descriptor.n = image.n;
  descriptor.h = image.h;
  descriptor.w = image.w;
  descriptor.c = image.c;
  descriptor.layout = image.layout;
  return descriptor;
}

unsigned int BlockCount(int64_t count) {
  const int64_t needed = 1 + (count - 1) / kThreadsPerBlock;
  return static_cast<unsigned int>(needed < kMaxBlocks ? needed : kMaxBlocks);
}

__device__ int64_t ThreadIndex() {
  return static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ int64_t ThreadStride() {
  return static_cast<int64_t>(blockDim.x) * gridDim.x;
}

__device__ int64_t StridedElementOffset(const DeviceTensor& tensor,
                                        int64_t linear_index) {
  int64_t offset = 0;
  for (int dim = tensor.rank - 1; dim >= 0; --dim) {
    const int64_t coordinate = linear_index % tensor.dims[dim];
    linear_index /= tensor.dims[dim];
    offset += coordinate * tensor.strides[dim];
  }
  return offset;
}

Result<CudaStream*> GetCudaStream(const OpContext& context) {
  Stream* stream = context.stream;
  if (stream == nullptr && context.context != nullptr) {
    stream = &context.context->stream();
  }
  if (stream == nullptr) {
    return Status::InvalidArgument(
        "CUDA kernel requires OpContext.stream or OpContext.context");
  }
  CudaStream* cuda_stream = dynamic_cast<CudaStream*>(stream);
  if (cuda_stream == nullptr) {
    return Status::InvalidArgument("CUDA kernel requires a CudaStream");
  }
  if (cuda_stream->handle() == nullptr) {
    return Status::BadState("CUDA stream is not initialized");
  }
  return cuda_stream;
}

Status FinishLaunch(cudaStream_t stream, const char* operation) {
  const Status launch_status = CudaStatus(cudaGetLastError(), operation);
  const Status synchronize_status =
      CudaStatus(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  if (!launch_status.ok()) return launch_status;
  return synchronize_status;
}

template <typename T>
__device__ T SaturatingCast(double value) {
  static_assert(std::is_integral<T>::value, "integral destination required");
  if (::isnan(value)) return static_cast<T>(0);
  const double low = static_cast<double>(std::numeric_limits<T>::lowest());
  const double high = static_cast<double>(std::numeric_limits<T>::max());
  if (value <= low) return std::numeric_limits<T>::lowest();
  if (value >= high) return std::numeric_limits<T>::max();
  return static_cast<T>(value);
}

template <typename T>
__device__ T SaturatingAdd(T lhs, T rhs) {
  if constexpr (std::is_signed<T>::value) {
    const T maximum = std::numeric_limits<T>::max();
    const T minimum = std::numeric_limits<T>::lowest();
    if (rhs > 0 && lhs > maximum - rhs) return maximum;
    if (rhs < 0 && lhs < minimum - rhs) return minimum;
  } else {
    const T maximum = std::numeric_limits<T>::max();
    if (lhs > maximum - rhs) return maximum;
  }
  return static_cast<T>(lhs + rhs);
}

template <typename T>
__device__ T SaturatingMultiply(T lhs, T rhs) {
  const T maximum = std::numeric_limits<T>::max();
  if constexpr (std::is_signed<T>::value) {
    const T minimum = std::numeric_limits<T>::lowest();
    if (lhs == 0 || rhs == 0) return static_cast<T>(0);
    if (lhs > 0) {
      if (rhs > 0) {
        if (lhs > maximum / rhs) return maximum;
      } else if (rhs < minimum / lhs) {
        return minimum;
      }
    } else {
      if (rhs > 0) {
        if (lhs < minimum / rhs) return minimum;
      } else if (rhs < maximum / lhs) {
        return maximum;
      }
    }
  } else {
    if (lhs != 0 && rhs > maximum / lhs) return maximum;
  }
  return static_cast<T>(lhs * rhs);
}

template <typename T>
__device__ T BinaryValue(T lhs, T rhs, bool multiply) {
  if constexpr (std::is_floating_point<T>::value) {
    return multiply ? static_cast<T>(lhs * rhs) : static_cast<T>(lhs + rhs);
  } else {
    return multiply ? SaturatingMultiply(lhs, rhs) : SaturatingAdd(lhs, rhs);
  }
}

template <typename T>
__global__ void BinaryKernel(DeviceTensor input, DeviceTensor other, T* output,
                             int64_t count, bool scalar_rhs, bool multiply) {
  const T* input_data = static_cast<const T*>(input.data);
  const T* other_data = static_cast<const T*>(other.data);
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    const T lhs = input_data[StridedElementOffset(input, index)];
    const T rhs = scalar_rhs ? other_data[0]
                             : other_data[StridedElementOffset(other, index)];
    output[index] = BinaryValue(lhs, rhs, multiply);
  }
}

__device__ double LoadValue(const void* data, DataType dtype) {
  switch (dtype) {
    case DataType::kUInt8:
      return static_cast<double>(*static_cast<const uint8_t*>(data));
    case DataType::kInt8:
      return static_cast<double>(*static_cast<const int8_t*>(data));
    case DataType::kUInt16:
      return static_cast<double>(*static_cast<const uint16_t*>(data));
    case DataType::kInt16:
      return static_cast<double>(*static_cast<const int16_t*>(data));
    case DataType::kFloat16:
      return static_cast<double>(
          __half2float(*static_cast<const __half*>(data)));
    case DataType::kFloat32:
      return static_cast<double>(*static_cast<const float*>(data));
    case DataType::kFloat64:
      return *static_cast<const double*>(data);
    case DataType::kInt32:
      return static_cast<double>(*static_cast<const int32_t*>(data));
    case DataType::kInt64:
      return static_cast<double>(*static_cast<const int64_t*>(data));
  }
  return 0.0;
}

__device__ void StoreValue(void* data, DataType dtype, double value) {
  switch (dtype) {
    case DataType::kUInt8:
      *static_cast<uint8_t*>(data) = SaturatingCast<uint8_t>(value);
      break;
    case DataType::kInt8:
      *static_cast<int8_t*>(data) = SaturatingCast<int8_t>(value);
      break;
    case DataType::kUInt16:
      *static_cast<uint16_t*>(data) = SaturatingCast<uint16_t>(value);
      break;
    case DataType::kInt16:
      *static_cast<int16_t*>(data) = SaturatingCast<int16_t>(value);
      break;
    case DataType::kFloat16:
      *static_cast<__half*>(data) = __float2half_rn(static_cast<float>(value));
      break;
    case DataType::kFloat32:
      *static_cast<float*>(data) = static_cast<float>(value);
      break;
    case DataType::kFloat64:
      *static_cast<double*>(data) = value;
      break;
    case DataType::kInt32:
      *static_cast<int32_t*>(data) = SaturatingCast<int32_t>(value);
      break;
    case DataType::kInt64:
      *static_cast<int64_t*>(data) = SaturatingCast<int64_t>(value);
      break;
  }
}

__global__ void CastKernel(DeviceTensor input, char* output, int64_t count,
                           size_t input_element_size,
                           size_t output_element_size, DataType input_dtype,
                           DataType output_dtype) {
  const char* input_data = static_cast<const char*>(input.data);
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    const size_t input_offset = static_cast<size_t>(
        StridedElementOffset(input, index) *
        static_cast<int64_t>(input_element_size));
    StoreValue(output + static_cast<size_t>(index) * output_element_size,
               output_dtype,
               LoadValue(input_data + input_offset, input_dtype));
  }
}

template <typename In, typename Out>
__global__ void NormalizeKernel(DeviceTensor input, Out* output, int64_t count,
                                int64_t channels, const double* parameters,
                                double epsilon) {
  const In* input_data = static_cast<const In*>(input.data);
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    const int64_t channel = index % channels;
    const Out value = static_cast<Out>(
        input_data[StridedElementOffset(input, index)]);
    const Out mean = static_cast<Out>(parameters[channel]);
    const Out denominator =
        static_cast<Out>(parameters[channels + channel] + epsilon);
    output[index] = (value - mean) / denominator;
  }
}

template <typename In, typename Out>
void LaunchNormalizeTyped(const DeviceTensor& input, Out* output, int64_t count,
                          int64_t channels, const double* parameters,
                          double epsilon, cudaStream_t stream) {
  NormalizeKernel<In, Out><<<BlockCount(count), kThreadsPerBlock, 0, stream>>>(
      input, output, count, channels, parameters, epsilon);
}

template <typename Out>
Status LaunchNormalize(const DeviceTensor& input, DataType input_dtype,
                       Out* output, int64_t count, int64_t channels,
                       const double* parameters, double epsilon,
                       cudaStream_t stream) {
  switch (input_dtype) {
    case DataType::kUInt8:
      LaunchNormalizeTyped<uint8_t>(input, output, count, channels, parameters,
                                    epsilon, stream);
      break;
    case DataType::kInt8:
      LaunchNormalizeTyped<int8_t>(input, output, count, channels, parameters,
                                   epsilon, stream);
      break;
    case DataType::kUInt16:
      LaunchNormalizeTyped<uint16_t>(input, output, count, channels, parameters,
                                     epsilon, stream);
      break;
    case DataType::kInt16:
      LaunchNormalizeTyped<int16_t>(input, output, count, channels, parameters,
                                    epsilon, stream);
      break;
    case DataType::kFloat32:
      LaunchNormalizeTyped<float>(input, output, count, channels, parameters,
                                  epsilon, stream);
      break;
    case DataType::kFloat64:
      LaunchNormalizeTyped<double>(input, output, count, channels, parameters,
                                   epsilon, stream);
      break;
    case DataType::kInt32:
      LaunchNormalizeTyped<int32_t>(input, output, count, channels, parameters,
                                    epsilon, stream);
      break;
    case DataType::kInt64:
      LaunchNormalizeTyped<int64_t>(input, output, count, channels, parameters,
                                    epsilon, stream);
      break;
    case DataType::kFloat16:
      return Status::Unsupported("CUDA Normalize does not support f16 input");
  }
  return FinishLaunch(stream, "CUDA Normalize kernel launch");
}

__device__ void DecodeImageIndex(const DeviceImage& image, int64_t linear_index,
                                 int64_t& n, int64_t& y, int64_t& x, int64_t& c) {
  if (image.layout == ImageLayout::kNHWC) {
    c = linear_index % image.c;
    linear_index /= image.c;
    x = linear_index % image.w;
    linear_index /= image.w;
    y = linear_index % image.h;
    n = linear_index / image.h;
  } else {
    x = linear_index % image.w;
    linear_index /= image.w;
    y = linear_index % image.h;
    linear_index /= image.h;
    c = linear_index % image.c;
    n = linear_index / image.c;
  }
}

__device__ int64_t ImageElementOffset(const DeviceTensor& tensor,
                                      const DeviceImage& image, int64_t n,
                                      int64_t y, int64_t x, int64_t c) {
  int64_t coordinates[4] = {};
  if (image.rank == 2) {
    coordinates[0] = y;
    coordinates[1] = x;
  } else if (image.rank == 3 && image.layout == ImageLayout::kNHWC) {
    coordinates[0] = y;
    coordinates[1] = x;
    coordinates[2] = c;
  } else if (image.rank == 3) {
    coordinates[0] = c;
    coordinates[1] = y;
    coordinates[2] = x;
  } else if (image.layout == ImageLayout::kNHWC) {
    coordinates[0] = n;
    coordinates[1] = y;
    coordinates[2] = x;
    coordinates[3] = c;
  } else {
    coordinates[0] = n;
    coordinates[1] = c;
    coordinates[2] = y;
    coordinates[3] = x;
  }
  int64_t offset = 0;
  for (int dim = 0; dim < image.rank; ++dim) {
    offset += coordinates[dim] * tensor.strides[dim];
  }
  return offset;
}

template <typename T>
__device__ void StoreImageValue(T* output, int64_t index, double value) {
  if constexpr (std::is_same<T, uint8_t>::value) {
    output[index] = SaturatingCast<uint8_t>(::round(value));
  } else {
    output[index] = static_cast<T>(value);
  }
}

template <typename T>
__global__ void ResizeKernel(DeviceTensor input, DeviceImage input_image,
                             DeviceImage output_image, T* output, int64_t count,
                             InterpolationMode interpolation) {
  const T* input_data = static_cast<const T*>(input.data);
  const double scale_y = static_cast<double>(input_image.h) / output_image.h;
  const double scale_x = static_cast<double>(input_image.w) / output_image.w;
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    int64_t n = 0;
    int64_t y = 0;
    int64_t x = 0;
    int64_t c = 0;
    DecodeImageIndex(output_image, index, n, y, x, c);
    double value = 0.0;
    if (interpolation == InterpolationMode::kNearest) {
      int64_t source_y = static_cast<int64_t>(y * scale_y);
      int64_t source_x = static_cast<int64_t>(x * scale_x);
      source_y = source_y < input_image.h ? source_y : input_image.h - 1;
      source_x = source_x < input_image.w ? source_x : input_image.w - 1;
      value = static_cast<double>(input_data[ImageElementOffset(
          input, input_image, n, source_y, source_x, c)]);
    } else {
      const double source_y = (static_cast<double>(y) + 0.5) * scale_y - 0.5;
      const double source_x = (static_cast<double>(x) + 0.5) * scale_x - 0.5;
      const double clamped_y = source_y > 0.0 ? source_y : 0.0;
      const double clamped_x = source_x > 0.0 ? source_x : 0.0;
      const int64_t y0 = static_cast<int64_t>(::floor(clamped_y));
      const int64_t x0 = static_cast<int64_t>(::floor(clamped_x));
      const int64_t y1 = y0 + 1 < input_image.h ? y0 + 1 : input_image.h - 1;
      const int64_t x1 = x0 + 1 < input_image.w ? x0 + 1 : input_image.w - 1;
      const double wy = clamped_y - y0;
      const double wx = clamped_x - x0;
      const double top_left = static_cast<double>(input_data[ImageElementOffset(
          input, input_image, n, y0, x0, c)]);
      const double top_right = static_cast<double>(input_data[ImageElementOffset(
          input, input_image, n, y0, x1, c)]);
      const double bottom_left = static_cast<double>(input_data[ImageElementOffset(
          input, input_image, n, y1, x0, c)]);
      const double bottom_right = static_cast<double>(input_data[ImageElementOffset(
          input, input_image, n, y1, x1, c)]);
      const double top = top_left + (top_right - top_left) * wx;
      const double bottom = bottom_left + (bottom_right - bottom_left) * wx;
      value = top + (bottom - top) * wy;
    }
    StoreImageValue(output, index, value);
  }
}

__global__ void CropKernel(DeviceTensor input, DeviceImage input_image,
                           DeviceImage output_image, char* output,
                           int64_t count, int64_t top, int64_t left,
                           size_t element_size) {
  const char* input_data = static_cast<const char*>(input.data);
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    int64_t n = 0;
    int64_t y = 0;
    int64_t x = 0;
    int64_t c = 0;
    DecodeImageIndex(output_image, index, n, y, x, c);
    const int64_t input_offset = ImageElementOffset(
        input, input_image, n, top + y, left + x, c);
    const char* source = input_data +
        static_cast<size_t>(input_offset * static_cast<int64_t>(element_size));
    char* destination = output + static_cast<size_t>(index) * element_size;
    for (size_t byte = 0; byte < element_size; ++byte) {
      destination[byte] = source[byte];
    }
  }
}

__global__ void TransposeKernel(DeviceTensor permuted_input, char* output,
                                int64_t count, size_t element_size) {
  const char* input_data = static_cast<const char*>(permuted_input.data);
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    const int64_t input_offset = StridedElementOffset(permuted_input, index);
    const char* source = input_data +
        static_cast<size_t>(input_offset * static_cast<int64_t>(element_size));
    char* destination = output + static_cast<size_t>(index) * element_size;
    for (size_t byte = 0; byte < element_size; ++byte) {
      destination[byte] = source[byte];
    }
  }
}

template <typename T>
__device__ double WarpSample(const DeviceTensor& input,
                             const DeviceImage& image, int64_t n, int64_t y,
                             int64_t x, int64_t c, double border_value) {
  if (y < 0 || y >= image.h || x < 0 || x >= image.w) return border_value;
  const T* input_data = static_cast<const T*>(input.data);
  return static_cast<double>(
      input_data[ImageElementOffset(input, image, n, y, x, c)]);
}

template <typename T>
__global__ void WarpAffineKernel(DeviceTensor input, DeviceImage input_image,
                                 DeviceImage output_image, DeviceAffine matrix,
                                 T* output, int64_t count,
                                 InterpolationMode interpolation,
                                 double border_value) {
  const double* m = matrix.values;
  const double determinant = m[0] * m[4] - m[1] * m[3];
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    int64_t n = 0;
    int64_t y = 0;
    int64_t x = 0;
    int64_t c = 0;
    DecodeImageIndex(output_image, index, n, y, x, c);
    const double dx = static_cast<double>(x) - m[2];
    const double dy = static_cast<double>(y) - m[5];
    const double source_x = (m[4] * dx - m[1] * dy) / determinant;
    const double source_y = (-m[3] * dx + m[0] * dy) / determinant;
    double value = border_value;
    if (interpolation == InterpolationMode::kNearest) {
      const int64_t source_y_rounded = ::llround(source_y);
      const int64_t source_x_rounded = ::llround(source_x);
      value = WarpSample<T>(input, input_image, n, source_y_rounded,
                            source_x_rounded, c, border_value);
    } else {
      const int64_t x0 = static_cast<int64_t>(::floor(source_x));
      const int64_t y0 = static_cast<int64_t>(::floor(source_y));
      const double wx = source_x - x0;
      const double wy = source_y - y0;
      const double top_left =
          WarpSample<T>(input, input_image, n, y0, x0, c, border_value);
      const double top_right =
          WarpSample<T>(input, input_image, n, y0, x0 + 1, c, border_value);
      const double bottom_left =
          WarpSample<T>(input, input_image, n, y0 + 1, x0, c, border_value);
      const double bottom_right =
          WarpSample<T>(input, input_image, n, y0 + 1, x0 + 1, c, border_value);
      const double top = top_left + (top_right - top_left) * wx;
      const double bottom = bottom_left + (bottom_right - bottom_left) * wx;
      value = top + (bottom - top) * wy;
    }
    StoreImageValue(output, index, value);
  }
}

Status CudaBinary(const OpContext& context, const Tensor& input,
                  const Tensor* other, Tensor* output, bool multiply) {
  auto stream_result = GetCudaStream(context);
  if (!stream_result.ok()) return stream_result.status();
  cudaStream_t stream = stream_result.value()->handle();
  const DeviceTensor input_descriptor = Describe(input);
  const DeviceTensor other_descriptor = Describe(*other);
  const int64_t count = input.Numel();
  const bool scalar_rhs = other->Numel() == 1;
  const unsigned int blocks = BlockCount(count);
  switch (input.dtype()) {
    case DataType::kUInt8:
      BinaryKernel<uint8_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input_descriptor, other_descriptor, output->Data<uint8_t>(), count,
          scalar_rhs, multiply);
      break;
    case DataType::kInt8:
      BinaryKernel<int8_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input_descriptor, other_descriptor, output->Data<int8_t>(), count,
          scalar_rhs, multiply);
      break;
    case DataType::kUInt16:
      BinaryKernel<uint16_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input_descriptor, other_descriptor, output->Data<uint16_t>(), count,
          scalar_rhs, multiply);
      break;
    case DataType::kInt16:
      BinaryKernel<int16_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input_descriptor, other_descriptor, output->Data<int16_t>(), count,
          scalar_rhs, multiply);
      break;
    case DataType::kFloat32:
      BinaryKernel<float><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input_descriptor, other_descriptor, output->Data<float>(), count,
          scalar_rhs, multiply);
      break;
    case DataType::kFloat64:
      BinaryKernel<double><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input_descriptor, other_descriptor, output->Data<double>(), count,
          scalar_rhs, multiply);
      break;
    case DataType::kInt32:
      BinaryKernel<int32_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input_descriptor, other_descriptor, output->Data<int32_t>(), count,
          scalar_rhs, multiply);
      break;
    case DataType::kInt64:
      BinaryKernel<int64_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input_descriptor, other_descriptor, output->Data<int64_t>(), count,
          scalar_rhs, multiply);
      break;
    case DataType::kFloat16:
      return Status::Unsupported("CUDA Add/Mul do not support f16");
  }
  return FinishLaunch(stream, multiply ? "CUDA Mul kernel launch"
                                       : "CUDA Add kernel launch");
}

Status CudaAdd(const OpContext& context, const Tensor& input,
               const Tensor* other, const void*, Tensor* output) {
  return CudaBinary(context, input, other, output, false);
}

Status CudaMul(const OpContext& context, const Tensor& input,
               const Tensor* other, const void*, Tensor* output) {
  return CudaBinary(context, input, other, output, true);
}

Status CudaCast(const OpContext& context, const Tensor& input, const Tensor*,
                const void*, Tensor* output) {
  auto stream_result = GetCudaStream(context);
  if (!stream_result.ok()) return stream_result.status();
  cudaStream_t stream = stream_result.value()->handle();
  const int64_t count = input.Numel();
  CastKernel<<<BlockCount(count), kThreadsPerBlock, 0, stream>>>(
      Describe(input), static_cast<char*>(output->data()), count,
      input.ElementSize(), output->ElementSize(), input.dtype(), output->dtype());
  return FinishLaunch(stream, "CUDA Cast kernel launch");
}

Status CudaNormalize(const OpContext& context, const Tensor& input,
                     const Tensor*, const void* params, Tensor* output) {
  auto stream_result = GetCudaStream(context);
  if (!stream_result.ok()) return stream_result.status();
  cudaStream_t stream = stream_result.value()->handle();
  const auto& options = *static_cast<const NormalizeParams*>(params);
  const int64_t channels = input.shape().Dim(input.shape().Rank() - 1);
  const size_t parameter_bytes = static_cast<size_t>(channels) * sizeof(double);
  void* parameter_memory = nullptr;
  Status status = CudaStatus(cudaMalloc(&parameter_memory, parameter_bytes * 2),
                             "cudaMalloc Normalize parameters");
  if (!status.ok()) return status;
  double* device_parameters = static_cast<double*>(parameter_memory);
  status = CudaStatus(
      cudaMemcpy(device_parameters, options.mean->data(), parameter_bytes,
                 cudaMemcpyHostToDevice),
      "cudaMemcpy Normalize mean");
  if (status.ok()) {
    status = CudaStatus(
        cudaMemcpy(device_parameters + channels, options.std->data(),
                   parameter_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy Normalize std");
  }
  if (!status.ok()) {
    (void)cudaFree(parameter_memory);
    return status;
  }

  const DeviceTensor input_descriptor = Describe(input);
  const int64_t count = input.Numel();
  if (output->dtype() == DataType::kFloat32) {
    status = LaunchNormalize(input_descriptor, input.dtype(),
                             output->Data<float>(), count, channels,
                             device_parameters, options.epsilon, stream);
  } else if (output->dtype() == DataType::kFloat64) {
    status = LaunchNormalize(input_descriptor, input.dtype(),
                             output->Data<double>(), count, channels,
                             device_parameters, options.epsilon, stream);
  } else {
    status = Status::Unsupported("CUDA Normalize output must be f32 or f64");
  }
  const Status free_status =
      CudaStatus(cudaFree(parameter_memory), "cudaFree Normalize parameters");
  if (!status.ok()) return status;
  return free_status;
}

Status LaunchResize(const DeviceTensor& input, const DeviceImage& input_image,
                    const DeviceImage& output_image, DataType dtype,
                    void* output, int64_t count,
                    InterpolationMode interpolation, cudaStream_t stream) {
  const unsigned int blocks = BlockCount(count);
  switch (dtype) {
    case DataType::kUInt8:
      ResizeKernel<uint8_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input, input_image, output_image, static_cast<uint8_t*>(output), count,
          interpolation);
      break;
    case DataType::kFloat32:
      ResizeKernel<float><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input, input_image, output_image, static_cast<float*>(output), count,
          interpolation);
      break;
    case DataType::kFloat64:
      ResizeKernel<double><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input, input_image, output_image, static_cast<double*>(output), count,
          interpolation);
      break;
    default:
      return Status::Unsupported("CUDA Resize supports u8, f32 and f64 only");
  }
  return FinishLaunch(stream, "CUDA Resize kernel launch");
}

Status CudaResize(const OpContext& context, const Tensor& input, const Tensor*,
                  const void* params, Tensor* output) {
  auto stream_result = GetCudaStream(context);
  if (!stream_result.ok()) return stream_result.status();
  const auto& options = *static_cast<const ResizeParams*>(params);
  auto image_result = ParseImage(input.shape(), options.layout);
  if (!image_result.ok()) return image_result.status();
  ImageInfo output_info = image_result.value();
  output_info.h = options.height;
  output_info.w = options.width;
  return LaunchResize(Describe(input), Describe(image_result.value()),
                      Describe(output_info), input.dtype(), output->data(),
                      output->Numel(), options.interpolation,
                      stream_result.value()->handle());
}

Status CudaCrop(const OpContext& context, const Tensor& input, const Tensor*,
                const void* params, Tensor* output) {
  auto stream_result = GetCudaStream(context);
  if (!stream_result.ok()) return stream_result.status();
  const auto& options = *static_cast<const CropParams*>(params);
  auto image_result = ParseImage(input.shape(), options.layout);
  if (!image_result.ok()) return image_result.status();
  ImageInfo output_info = image_result.value();
  output_info.h = options.height;
  output_info.w = options.width;
  const int64_t count = output->Numel();
  cudaStream_t stream = stream_result.value()->handle();
  CropKernel<<<BlockCount(count), kThreadsPerBlock, 0, stream>>>(
      Describe(input), Describe(image_result.value()), Describe(output_info),
      static_cast<char*>(output->data()), count, options.top, options.left,
      input.ElementSize());
  return FinishLaunch(stream, "CUDA Crop kernel launch");
}

Status CudaTranspose(const OpContext& context, const Tensor& input,
                     const Tensor*, const void* params, Tensor* output) {
  auto stream_result = GetCudaStream(context);
  if (!stream_result.ok()) return stream_result.status();
  const auto& options = *static_cast<const TransposeParams*>(params);
  const DeviceTensor input_descriptor = Describe(input);
  DeviceTensor permuted = input_descriptor;
  for (int dim = 0; dim < input_descriptor.rank; ++dim) {
    const int source_dim = (*options.order)[static_cast<size_t>(dim)];
    permuted.dims[dim] = input_descriptor.dims[source_dim];
    permuted.strides[dim] = input_descriptor.strides[source_dim];
  }
  const int64_t count = output->Numel();
  cudaStream_t stream = stream_result.value()->handle();
  TransposeKernel<<<BlockCount(count), kThreadsPerBlock, 0, stream>>>(
      permuted, static_cast<char*>(output->data()), count, input.ElementSize());
  return FinishLaunch(stream, "CUDA Transpose kernel launch");
}

Status LaunchWarpAffine(const DeviceTensor& input,
                        const DeviceImage& input_image,
                        const DeviceImage& output_image, DataType dtype,
                        const DeviceAffine& matrix, void* output, int64_t count,
                        InterpolationMode interpolation, double border_value,
                        cudaStream_t stream) {
  const unsigned int blocks = BlockCount(count);
  switch (dtype) {
    case DataType::kUInt8:
      WarpAffineKernel<uint8_t><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input, input_image, output_image, matrix,
          static_cast<uint8_t*>(output), count, interpolation, border_value);
      break;
    case DataType::kFloat32:
      WarpAffineKernel<float><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input, input_image, output_image, matrix, static_cast<float*>(output),
          count, interpolation, border_value);
      break;
    case DataType::kFloat64:
      WarpAffineKernel<double><<<blocks, kThreadsPerBlock, 0, stream>>>(
          input, input_image, output_image, matrix, static_cast<double*>(output),
          count, interpolation, border_value);
      break;
    default:
      return Status::Unsupported(
          "CUDA WarpAffine supports u8, f32 and f64 only");
  }
  return FinishLaunch(stream, "CUDA WarpAffine kernel launch");
}

Status CudaWarpAffine(const OpContext& context, const Tensor& input,
                      const Tensor*, const void* params, Tensor* output) {
  auto stream_result = GetCudaStream(context);
  if (!stream_result.ok()) return stream_result.status();
  const auto& options = *static_cast<const WarpAffineParams*>(params);
  auto image_result = ParseImage(input.shape(), options.layout);
  if (!image_result.ok()) return image_result.status();
  ImageInfo output_info = image_result.value();
  output_info.h = options.height;
  output_info.w = options.width;
  DeviceAffine matrix = {};
  for (size_t index = 0; index < options.matrix->size(); ++index) {
    matrix.values[index] = (*options.matrix)[index];
  }
  return LaunchWarpAffine(
      Describe(input), Describe(image_result.value()), Describe(output_info),
      input.dtype(), matrix, output->data(), output->Numel(),
      options.interpolation, options.border_value,
      stream_result.value()->handle());
}

}  // namespace

void RegisterCudaKernels() {
  const BackendType backend = BackendType::kCUDA;
  const CpuKernelLevel level = CpuKernelLevel::kNaive;
  RegisterKernel(OpKind::kAdd, backend, level, &CudaAdd);
  RegisterKernel(OpKind::kMul, backend, level, &CudaMul);
  RegisterKernel(OpKind::kCast, backend, level, &CudaCast);
  RegisterKernel(OpKind::kNormalize, backend, level, &CudaNormalize);
  RegisterKernel(OpKind::kResize, backend, level, &CudaResize);
  RegisterKernel(OpKind::kCrop, backend, level, &CudaCrop);
  RegisterKernel(OpKind::kTranspose, backend, level, &CudaTranspose);
  RegisterKernel(OpKind::kWarpAffine, backend, level, &CudaWarpAffine);
}

}  // namespace ops_internal
}  // namespace mycv
