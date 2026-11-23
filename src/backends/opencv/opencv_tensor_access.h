#ifndef hpcplatform_SRC_BACKENDS_OPENCV_OPENCV_TENSOR_ACCESS_H_
#define hpcplatform_SRC_BACKENDS_OPENCV_OPENCV_TENSOR_ACCESS_H_

// Backend service header for operator OpenCV implementations
// (docs/architecture.md §6.2): dtype/contiguity checks, cv::Mat views over
// Tensor memory, and the exception bridge. Operators include this header; it
// is never exposed through the public API.

#include <functional>

#include <opencv2/core.hpp>

#include "framework/image_utils.h"
#include "hpcplatform/core/error.h"
#include "hpcplatform/core/tensor.h"

namespace hpcplatform {
namespace ops_internal {

// CV depth code for a dtype, or -1 when the dtype has no OpenCV equivalent.
int CvDepth(DataType dtype);

// OpenCV kernels accept u8/f32/f64, contiguous tensors and at most INT_MAX
// elements (cv::Mat dimensions are int).
Status CheckDType(const Tensor& tensor);

// Runs an OpenCV call, bridging cv::Exception into the Status error model
// (hpcplatform is built with -fno-exceptions; OpenCV is not).
Status CvCall(const std::function<void()>& fn);

// Writable view over the tensor's memory.
void* CvData(const Tensor& tensor);

// 1-D cv::Mat view over the whole tensor (elementwise ops).
cv::Mat FlatMat(const Tensor& tensor);

// H x W x C cv::Mat view over the n-th image of an image tensor. Validates
// the tensor via CheckDType; returns kUnsupported/kInvalidArgument otherwise.
Result<cv::Mat> ImageMat(const Tensor& tensor, const ImageInfo& info,
                         int64_t n);

// OpenCV image kernels take HW/HWC/NHWC only; NCHW needs an explicit
// Transpose first.
Status RejectNchw(ImageLayout layout);

}  // namespace ops_internal
}  // namespace hpcplatform

#endif  // hpcplatform_SRC_BACKENDS_OPENCV_OPENCV_TENSOR_ACCESS_H_
