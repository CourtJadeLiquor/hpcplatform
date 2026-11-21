// resize 的 OpenCV 后端：按 batch 为每张图建立 HxWxC 的 cv::Mat 视图，
// 直接调用 cv::resize 完成缩放，本实现不含自写插值公式。
//
// 插值标志：最近邻 -> cv::INTER_NEAREST，双线性 -> cv::INTER_LINEAR；
// 目标尺寸以 cv::Size(width, height) 传入，fx/fy 置 0（由 dsize 决定）。
// 布局/通道：OpenCV 只接受 HW/HWC/NHWC，NCHW 由 RejectNchw 拒绝，需先
// 显式转置；ImageMat 只建立 header，不拷贝数据。
//
// 位精确性：cv::resize 使用 OpenCV 自身的定点/向量化路径与边界处理，
// 与手写 CPU/CUDA 核只在容差意义下一致，不保证逐位相同。
// 异常：cv::Exception 由 CvCall 捕获并转换为 Status。

#include <opencv2/imgproc.hpp>

#include "backends/opencv/opencv_tensor_access.h"
#include "framework/image_utils.h"
#include "mycv/ops/image.h"
#include "resize_internal.h"

namespace mycv {
namespace ops_internal {

// OpenCV 后端入口：输入 input、参数 options（目标高宽/插值/布局），
// 输出 output。步骤：拒绝 NCHW -> 解析输入 ImageInfo -> 覆盖输出 h/w
// -> 逐 batch 建立源/目标 Mat 并调用 cv::resize。
// 复杂度由 OpenCV 实现决定，量级为 O(N*H_out*W_out*C)。
Status ResizeOpenCv(const OpContext&, const Tensor& input, const Tensor*,
                    const void* params, Tensor* output) {
  const auto& options = *static_cast<const ResizeParams*>(params);
  MYCV_RETURN_IF_ERROR(RejectNchw(options.layout));
  auto image = ParseImage(input.shape(), options.layout);
  if (!image.ok()) return image.status();
  ImageInfo output_info = image.value();
  // 输出与输入同 rank/layout，仅 H/W 不同。
  output_info.h = options.height;
  output_info.w = options.width;
  return CvCall([&] {
    for (int64_t n = 0; n < image.value().n; ++n) {
      // 第 n 张图的 HxWxC 视图；失败说明 dtype/连续性不满足 OpenCV 要求。
      auto source = ImageMat(input, image.value(), n);
      auto destination = ImageMat(*output, output_info, n);
      if (!source.ok() || !destination.ok()) {
        throw cv::Exception(0, "invalid image tensor", __func__, __FILE__,
                            __LINE__);
      }
      cv::resize(source.value(), destination.value(),
                 cv::Size(static_cast<int>(options.width),
                          static_cast<int>(options.height)),
                 0, 0,
                 // 两种插值一一映射到 OpenCV 的配套标志。
                 options.interpolation == InterpolationMode::kNearest
                     ? cv::INTER_NEAREST
                     : cv::INTER_LINEAR);
    }
  });
}

}  // namespace ops_internal
}  // namespace mycv
