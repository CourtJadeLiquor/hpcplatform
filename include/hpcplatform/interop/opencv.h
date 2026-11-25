#ifndef hpcplatform_INTEROP_OPENCV_H_
#define hpcplatform_INTEROP_OPENCV_H_

#include <opencv2/core.hpp>

#include "hpcplatform/core/context.h"
#include "hpcplatform/core/conversion.h"
#include "hpcplatform/core/error.h"
#include "hpcplatform/core/tensor.h"

namespace hpcplatform {

// TensorConvert specialization registering cv::Mat with Tensor::From/Into.
//
// Both directions are self-contained: the conversion owns the memory movement,
// so callers describe only the algorithm.
//   - From copies a 2D Mat into a Tensor on `context`. For a CPU/OpenCV context
//     the copy is host-to-host; for a device context the Mat is staged through a
//     borrowed host buffer and uploaded with the device backend (synchronous).
//     The Mat may be a non-contiguous ROI.
//   - Into copies a contiguous or strided HW/HWC Tensor into an owning cv::Mat.
//     A device Tensor is downloaded transparently; a non-contiguous device
//     tensor is materialized once with Clone() first.
template <>
struct TensorConvert<cv::Mat> {
  static Result<Tensor> From(const cv::Mat& mat, Context& context);
  static Result<cv::Mat> Into(const Tensor& tensor);
};

}  // namespace hpcplatform

#endif  // hpcplatform_INTEROP_OPENCV_H_
