#include <opencv2/imgproc.hpp>

#include "backends/opencv/opencv_tensor_access.h"
#include "framework/image_utils.h"
#include "hpcplatform/ops/image.h"
#include "gaussian_blur_internal.h"

// OpenCV backend: delegates Gaussian blur directly to cv::GaussianBlur.
// OpenCV also uses separable passes internally (horizontal and vertical 1D
// convolutions, each O(H*W*K), reducing direct 2D convolution's O(H*W*K^2));
// no custom loops are needed here. cv::BORDER_REPLICATE matches the CPU/CUDA
// replicate semantics. Supports HW/HWC/NHWC; RejectNchw rejects NCHW, requiring
// callers to Transpose explicitly. For each batch n, ImageMat passes a zero-copy
// H x W, C-channel cv::Mat view to OpenCV.
// Precision: OpenCV's internal paths (fixed-point for u8 and its own floating-
// point accumulation order) differ from the CPU levels' per-tap double
// accumulation, so bit exactness is not guaranteed. Compare across backends
// with 1e-4 relative tolerance (docs/qa/testing.md §4.2).
namespace hpcplatform {
namespace ops_internal {

// Entry point: reject NCHW and parse the image, then call cv::GaussianBlur for
// each batch inside CvCall. The kernel size is (K,K), sigma is used for both x
// and y, and the border mode is BORDER_REPLICATE. Source and destination are
// Mat views over the same tensor memory. CvCall catches cv::Exception from
// OpenCV and converts it to Status.
Status GaussianBlurOpenCv(const OpContext&, const Tensor& input, const Tensor*,
                          const void* params, Tensor* output) {
  const auto& options = *static_cast<const GaussianBlurParams*>(params);
  hpcplatform_RETURN_IF_ERROR(RejectNchw(options.layout));
  auto image = ParseImage(input.shape(), options.layout);
  if (!image.ok()) return image.status();
  return CvCall([&] {
    for (int64_t n = 0; n < image.value().n; ++n) {
      auto source = ImageMat(input, image.value(), n);
      auto destination = ImageMat(*output, image.value(), n);
      if (!source.ok() || !destination.ok()) {
        throw cv::Exception(0, "invalid image tensor", __func__, __FILE__,
                            __LINE__);
      }
      cv::GaussianBlur(source.value(), destination.value(),
                       cv::Size(options.kernel_size, options.kernel_size),
                       options.sigma, options.sigma, cv::BORDER_REPLICATE);
    }
  });
}

}  // namespace ops_internal
}  // namespace hpcplatform
