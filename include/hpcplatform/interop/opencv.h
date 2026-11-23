#ifndef hpcplatform_INTEROP_OPENCV_H_
#define hpcplatform_INTEROP_OPENCV_H_

#include <opencv2/core.hpp>

#include "hpcplatform/core/context.h"
#include "hpcplatform/core/error.h"
#include "hpcplatform/core/tensor.h"

namespace hpcplatform {

// Deep-copy a 2D cv::Mat into a contiguous HW/HWC host Tensor without retaining
// Mat storage.
//
// Deep-copy a 2D cv::Mat into a contiguous HW/HWC host Tensor. The Mat may be
// a non-contiguous ROI; its storage is never retained by the Tensor.
Result<Tensor> CopyFromCvMat(Context& context, const cv::Mat& mat);

// Deep-copy a contiguous HW/HWC host Tensor into a cv::Mat that owns its memory.
//
// Deep-copy a contiguous or strided HW/HWC host Tensor into an owning cv::Mat.
Result<cv::Mat> CopyToCvMat(const Tensor& tensor);

}  // namespace hpcplatform

#endif  // hpcplatform_INTEROP_OPENCV_H_
