#include "backends/opencv/opencv_tensor_access.h"

#include <limits>
#include <string>

#include <opencv2/core.hpp>

#include "mycv/core/dtype.h"

namespace mycv {
namespace ops_internal {

int CvDepth(DataType dtype) {
  switch (dtype) {
    case DataType::kUInt8:
      return CV_8U;
    case DataType::kFloat32:
      return CV_32F;
    case DataType::kFloat64:
      return CV_64F;
    default:
      return -1;
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
    return Status::Internal(std::string("OpenCV kernel failed: ") +
                            error.what());
  }
}

void* CvData(const Tensor& tensor) {
  return const_cast<void*>(tensor.data());
}

cv::Mat FlatMat(const Tensor& tensor) {
  return cv::Mat(1, static_cast<int>(tensor.Numel()), CvDepth(tensor.dtype()),
                 CvData(tensor));
}

Result<cv::Mat> ImageMat(const Tensor& tensor, const ImageInfo& info,
                         int64_t n) {
  MYCV_RETURN_IF_ERROR(CheckDType(tensor));
  if (info.c > CV_CN_MAX)
    return Status::Unsupported("too many channels for cv::Mat");
  const int64_t first = ImageLinearIndex(info, n, 0, 0, 0);
  char* data = static_cast<char*>(CvData(tensor)) +
               static_cast<size_t>(first) * tensor.ElementSize();
  return cv::Mat(static_cast<int>(info.h), static_cast<int>(info.w),
                 CV_MAKETYPE(CvDepth(tensor.dtype()), static_cast<int>(info.c)),
                 data);
}

Status RejectNchw(ImageLayout layout) {
  if (layout == ImageLayout::kNCHW) {
    return Status::InvalidArgument(
        "OpenCV backend supports HW/HWC/NHWC only; NCHW needs an explicit "
        "Transpose");
  }
  return Status::Ok();
}

}  // namespace ops_internal
}  // namespace mycv
