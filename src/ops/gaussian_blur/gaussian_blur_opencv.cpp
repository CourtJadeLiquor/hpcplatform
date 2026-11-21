#include <opencv2/imgproc.hpp>

#include "backends/opencv/opencv_tensor_access.h"
#include "framework/image_utils.h"
#include "mycv/ops/image.h"
#include "gaussian_blur_internal.h"

// OpenCV 后端：直接把高斯模糊委托给 cv::GaussianBlur。
// OpenCV 内部同样按可分离方式实现（水平 + 垂直两趟一维卷积，每趟
// O(H*W*K)，把二维直接卷积的 O(H*W*K^2) 降下来），这里不再手写循环。
// 边界模式选 cv::BORDER_REPLICATE，与 CPU/CUDA 的 replicate 语义一致。
// 支持 HW/HWC/NHWC：NCHW 由 RejectNchw 拒绝，需调用方显式 Transpose；
// 每个 batch n 通过 ImageMat 以零拷贝 cv::Mat 视图（H x W、C 通道）
// 交给 OpenCV。
// 精度：OpenCV 内部路径（u8 用定点、浮点用自己的累加顺序）与 CPU 层
// 的 double 逐 tap 累加不同，因此不保证位精确，跨后端按 1e-4 相对
// 容差比较（docs/qa/testing.md §4.2）。
namespace mycv {
namespace ops_internal {

// 入口：拒绝 NCHW、解析图像，然后在 CvCall 中按 batch 逐张调用
// cv::GaussianBlur——核大小为 (K,K)，x/y 两个方向都传 sigma，
// 边界 BORDER_REPLICATE；源/目标是同一张量内存上的 Mat 视图。
// OpenCV 抛出的 cv::Exception 由 CvCall 捕获并转为 Status。
Status GaussianBlurOpenCv(const OpContext&, const Tensor& input, const Tensor*,
                          const void* params, Tensor* output) {
  const auto& options = *static_cast<const GaussianBlurParams*>(params);
  MYCV_RETURN_IF_ERROR(RejectNchw(options.layout));
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
}  // namespace mycv
