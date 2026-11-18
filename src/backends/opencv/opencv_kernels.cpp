#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "core/value_cast.h"
#include "ops/image_utils.h"
#include "ops/op_params.h"
#include "ops/op_registry.h"

namespace mycv {
namespace ops_internal {
namespace {

int CvDepth(DataType dtype) {
  switch (dtype) {
    case DataType::kUInt8: return CV_8U;
    case DataType::kFloat32: return CV_32F;
    case DataType::kFloat64: return CV_64F;
    default: return -1;
  }
}

Status CheckDType(const Tensor& tensor) {
  if (CvDepth(tensor.dtype()) < 0) {
    return Status::Unsupported("OpenCV backend supports u8, f32 and f64 only");
  }
  if (!tensor.IsContiguous()) {
    return Status::InvalidArgument("OpenCV kernels require contiguous tensors");
  }
  if (tensor.Numel() > std::numeric_limits<int>::max()) {
    return Status::Unsupported("tensor is too large for cv::Mat dimensions");
  }
  return Status::Ok();
}

Status CvCall(const std::function<void()>& fn) {
  try {
    fn();
    return Status::Ok();
  } catch (const cv::Exception& error) {
    return Status::Internal(std::string("OpenCV kernel failed: ") + error.what());
  }
}

void* CvData(const Tensor& tensor) {
  return const_cast<void*>(tensor.data());
}

cv::Mat FlatMat(const Tensor& tensor) {
  return cv::Mat(1, static_cast<int>(tensor.Numel()), CvDepth(tensor.dtype()),
                 CvData(tensor));
}

Result<cv::Mat> ImageMat(const Tensor& tensor, const ImageInfo& info, int64_t n) {
  MYCV_RETURN_IF_ERROR(CheckDType(tensor));
  if (info.c > CV_CN_MAX) return Status::Unsupported("too many channels for cv::Mat");
  const int64_t first = ImageLinearIndex(info, n, 0, 0, 0);
  char* data = static_cast<char*>(CvData(tensor)) +
               static_cast<size_t>(first) * tensor.ElementSize();
  return cv::Mat(static_cast<int>(info.h), static_cast<int>(info.w),
                 CV_MAKETYPE(CvDepth(tensor.dtype()), static_cast<int>(info.c)), data);
}

Status RejectNchw(ImageLayout layout) {
  if (layout == ImageLayout::kNCHW) {
    return Status::InvalidArgument(
        "OpenCV backend supports HW/HWC/NHWC only; NCHW needs an explicit Transpose");
  }
  return Status::Ok();
}

Status OpenCvBinary(const Tensor& input, const Tensor* other, Tensor* output,
                    bool multiply) {
  MYCV_RETURN_IF_ERROR(CheckDType(input));
  MYCV_RETURN_IF_ERROR(CheckDType(*other));
  MYCV_RETURN_IF_ERROR(CheckDType(*output));
  return CvCall([&] {
    cv::Mat lhs = FlatMat(input);
    cv::Mat dst = FlatMat(*output);
    if (other->Numel() == 1) {
      const double scalar = static_cast<double>(
          detail::LoadAsLongDouble(other->data(), other->dtype()));
      if (multiply) {
        cv::multiply(lhs, cv::Scalar::all(scalar), dst);
      } else {
        cv::add(lhs, cv::Scalar::all(scalar), dst);
      }
    } else {
      cv::Mat rhs = FlatMat(*other);
      if (multiply) {
        cv::multiply(lhs, rhs, dst);
      } else {
        cv::add(lhs, rhs, dst);
      }
    }
  });
}

Status OpenCvAdd(const OpContext&, const Tensor& input, const Tensor* other,
                 const void*, Tensor* output) {
  return OpenCvBinary(input, other, output, false);
}

Status OpenCvMul(const OpContext&, const Tensor& input, const Tensor* other,
                 const void*, Tensor* output) {
  return OpenCvBinary(input, other, output, true);
}

Status OpenCvCast(const OpContext&, const Tensor& input, const Tensor*,
                  const void*, Tensor* output) {
  MYCV_RETURN_IF_ERROR(CheckDType(input));
  MYCV_RETURN_IF_ERROR(CheckDType(*output));
  if (output->dtype() == DataType::kUInt8) {
    const char* source = static_cast<const char*>(input.data());
    char* destination = static_cast<char*>(output->data());
    for (int64_t i = 0; i < input.Numel(); ++i) {
      detail::CastElement(source + static_cast<size_t>(i) * input.ElementSize(),
                          input.dtype(),
                          destination + static_cast<size_t>(i) * output->ElementSize(),
                          output->dtype());
    }
    return Status::Ok();
  }
  return CvCall([&] {
    cv::Mat source = FlatMat(input);
    cv::Mat destination = FlatMat(*output);
    source.convertTo(destination, CvDepth(output->dtype()));
  });
}

Status OpenCvNormalize(const OpContext&, const Tensor& input, const Tensor*,
                       const void* params, Tensor* output) {
  MYCV_RETURN_IF_ERROR(CheckDType(input));
  MYCV_RETURN_IF_ERROR(CheckDType(*output));
  const auto& options = *static_cast<const NormalizeParams*>(params);
  const int64_t channels = input.shape().Dim(input.shape().Rank() - 1);
  if (channels > CV_CN_MAX) return Status::Unsupported("too many channels for cv::Mat");
  return CvCall([&] {
    const int rows = static_cast<int>(input.Numel() / channels);
    cv::Mat source(rows, 1,
                   CV_MAKETYPE(CvDepth(input.dtype()), static_cast<int>(channels)),
                   CvData(input));
    cv::Mat converted;
    source.convertTo(converted, CV_MAKETYPE(CvDepth(output->dtype()),
                                            static_cast<int>(channels)));
    std::vector<cv::Mat> planes;
    cv::split(converted, planes);
    for (size_t c = 0; c < planes.size(); ++c) {
      cv::Mat shifted;
      cv::subtract(planes[c], cv::Scalar::all((*options.mean)[c]), shifted);
      cv::divide(shifted,
                 cv::Scalar::all((*options.std)[c] + options.epsilon), planes[c]);
    }
    cv::Mat destination(rows, 1,
                        CV_MAKETYPE(CvDepth(output->dtype()), static_cast<int>(channels)),
                        output->data());
    cv::merge(planes, destination);
  });
}

Status OpenCvResize(const OpContext&, const Tensor& input, const Tensor*,
                    const void* params, Tensor* output) {
  const auto& options = *static_cast<const ResizeParams*>(params);
  MYCV_RETURN_IF_ERROR(RejectNchw(options.layout));
  auto image = ParseImage(input.shape(), options.layout);
  if (!image.ok()) return image.status();
  ImageInfo output_info = image.value();
  output_info.h = options.height;
  output_info.w = options.width;
  return CvCall([&] {
    for (int64_t n = 0; n < image.value().n; ++n) {
      auto source = ImageMat(input, image.value(), n);
      auto destination = ImageMat(*output, output_info, n);
      if (!source.ok() || !destination.ok()) {
        throw cv::Exception(0, "invalid image tensor", __func__, __FILE__, __LINE__);
      }
      cv::resize(source.value(), destination.value(),
                 cv::Size(static_cast<int>(options.width),
                          static_cast<int>(options.height)),
                 0, 0,
                 options.interpolation == InterpolationMode::kNearest
                     ? cv::INTER_NEAREST
                     : cv::INTER_LINEAR);
    }
  });
}

Status OpenCvCrop(const OpContext&, const Tensor& input, const Tensor*,
                  const void* params, Tensor* output) {
  const auto& options = *static_cast<const CropParams*>(params);
  MYCV_RETURN_IF_ERROR(RejectNchw(options.layout));
  auto image = ParseImage(input.shape(), options.layout);
  if (!image.ok()) return image.status();
  ImageInfo output_info = image.value();
  output_info.h = options.height;
  output_info.w = options.width;
  return CvCall([&] {
    for (int64_t n = 0; n < image.value().n; ++n) {
      auto source = ImageMat(input, image.value(), n);
      auto destination = ImageMat(*output, output_info, n);
      if (!source.ok() || !destination.ok()) {
        throw cv::Exception(0, "invalid image tensor", __func__, __FILE__, __LINE__);
      }
      source.value()(cv::Rect(static_cast<int>(options.left),
                              static_cast<int>(options.top),
                              static_cast<int>(options.width),
                              static_cast<int>(options.height)))
          .copyTo(destination.value());
    }
  });
}

Status OpenCvTranspose(const OpContext&, const Tensor& input, const Tensor*,
                       const void* params, Tensor* output) {
  MYCV_RETURN_IF_ERROR(CheckDType(input));
  MYCV_RETURN_IF_ERROR(CheckDType(*output));
  const auto& order = *static_cast<const TransposeParams*>(params)->order;
  return CvCall([&] {
    if (input.shape().Rank() == 2 && order == std::vector<int>{1, 0}) {
      cv::Mat source(static_cast<int>(input.shape().Dim(0)),
                     static_cast<int>(input.shape().Dim(1)), CvDepth(input.dtype()),
                     CvData(input));
      cv::Mat destination(static_cast<int>(output->shape().Dim(0)),
                          static_cast<int>(output->shape().Dim(1)),
                          CvDepth(output->dtype()), output->data());
      cv::transpose(source, destination);
      return;
    }
    if (input.shape().Rank() == 3 && order == std::vector<int>{1, 0, 2}) {
      cv::Mat source(static_cast<int>(input.shape().Dim(0)),
                     static_cast<int>(input.shape().Dim(1)),
                     CV_MAKETYPE(CvDepth(input.dtype()),
                                 static_cast<int>(input.shape().Dim(2))),
                     CvData(input));
      cv::Mat destination(static_cast<int>(output->shape().Dim(0)),
                          static_cast<int>(output->shape().Dim(1)),
                          CV_MAKETYPE(CvDepth(output->dtype()),
                                      static_cast<int>(output->shape().Dim(2))),
                          output->data());
      cv::transpose(source, destination);
      return;
    }

    const bool hwc_to_chw = input.shape().Rank() == 3 &&
                            order == std::vector<int>{2, 0, 1};
    const bool nhwc_to_nchw = input.shape().Rank() == 4 &&
                              order == std::vector<int>{0, 3, 1, 2};
    if (!hwc_to_chw && !nhwc_to_nchw) {
      throw cv::Exception(0, "unsupported permutation", __func__, __FILE__, __LINE__);
    }
    auto image = ParseImage(input.shape(), ImageLayout::kNHWC);
    if (!image.ok()) {
      throw cv::Exception(0, "invalid NHWC input", __func__, __FILE__, __LINE__);
    }
    for (int64_t n = 0; n < image.value().n; ++n) {
      auto source = ImageMat(input, image.value(), n);
      if (!source.ok()) {
        throw cv::Exception(0, "invalid image", __func__, __FILE__, __LINE__);
      }
      std::vector<cv::Mat> planes;
      cv::split(source.value(), planes);
      for (int64_t c = 0; c < image.value().c; ++c) {
        const int64_t plane = n * image.value().c + c;
        char* destination_data = static_cast<char*>(output->data()) +
            static_cast<size_t>(plane * image.value().h * image.value().w) *
                output->ElementSize();
        cv::Mat destination(static_cast<int>(image.value().h),
                            static_cast<int>(image.value().w),
                            CvDepth(output->dtype()), destination_data);
        planes[static_cast<size_t>(c)].copyTo(destination);
      }
    }
  });
}

Status OpenCvWarpAffine(const OpContext&, const Tensor& input, const Tensor*,
                        const void* params, Tensor* output) {
  const auto& options = *static_cast<const WarpAffineParams*>(params);
  MYCV_RETURN_IF_ERROR(RejectNchw(options.layout));
  auto image = ParseImage(input.shape(), options.layout);
  if (!image.ok()) return image.status();
  ImageInfo output_info = image.value();
  output_info.h = options.height;
  output_info.w = options.width;
  return CvCall([&] {
    const cv::Mat matrix(2, 3, CV_64F,
                         const_cast<double*>(options.matrix->data()));
    for (int64_t n = 0; n < image.value().n; ++n) {
      auto source = ImageMat(input, image.value(), n);
      auto destination = ImageMat(*output, output_info, n);
      if (!source.ok() || !destination.ok()) {
        throw cv::Exception(0, "invalid image tensor", __func__, __FILE__, __LINE__);
      }
      cv::warpAffine(source.value(), destination.value(), matrix,
                     cv::Size(static_cast<int>(options.width),
                              static_cast<int>(options.height)),
                     options.interpolation == InterpolationMode::kNearest
                         ? cv::INTER_NEAREST
                         : cv::INTER_LINEAR,
                     cv::BORDER_CONSTANT,
                     cv::Scalar::all(options.border_value));
    }
  });
}

}  // namespace

void RegisterOpenCvKernels() {
  const BackendType backend = BackendType::kOpenCV;
  const CpuKernelLevel level = CpuKernelLevel::kNaive;
  RegisterKernel(OpKind::kAdd, backend, level, &OpenCvAdd);
  RegisterKernel(OpKind::kMul, backend, level, &OpenCvMul);
  RegisterKernel(OpKind::kCast, backend, level, &OpenCvCast);
  RegisterKernel(OpKind::kNormalize, backend, level, &OpenCvNormalize);
  RegisterKernel(OpKind::kResize, backend, level, &OpenCvResize);
  RegisterKernel(OpKind::kCrop, backend, level, &OpenCvCrop);
  RegisterKernel(OpKind::kTranspose, backend, level, &OpenCvTranspose);
  RegisterKernel(OpKind::kWarpAffine, backend, level, &OpenCvWarpAffine);
}

}  // namespace ops_internal
}  // namespace mycv
