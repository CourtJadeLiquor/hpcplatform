// OpenCV backend for resize: creates an HxWxC cv::Mat view for each image in
// the batch and calls cv::resize directly; no custom interpolation formulas.
//
// Interpolation flags: nearest -> cv::INTER_NEAREST; bilinear -> cv::INTER_LINEAR.
// The target size is passed as cv::Size(width, height), with fx/fy set to 0
// (dsize determines the scaling). Layout/channels: OpenCV accepts only
// HW/HWC/NHWC; RejectNchw rejects NCHW, which requires an explicit transpose.
// ImageMat creates only a header and does not copy data.
//
// Bit exactness: cv::resize uses OpenCV's fixed-point/vectorized paths and
// border handling, so it agrees with the custom CPU/CUDA kernels only within
// tolerance, not bit for bit. Exceptions: CvCall catches cv::Exception and
// converts it to Status.

#include <opencv2/imgproc.hpp>

#include "backends/opencv/opencv_tensor_access.h"
#include "framework/image_utils.h"
#include "hpcplatform/ops/image.h"
#include "resize_internal.h"

namespace hpcplatform
{
    namespace ops_internal
    {

        // OpenCV backend entry point: input is the source tensor, options contains the
        // target height/width, interpolation, and layout, and output is the destination.
        // Steps: reject NCHW -> parse input ImageInfo -> override output h/w -> create
        // source/destination Mat views per batch and call cv::resize. Complexity depends
        // on OpenCV and is on the order of O(N*H_out*W_out*C).
        Status ResizeOpenCv(const OpContext&,
                            const Tensor& input,
                            const Tensor*,
                            const void* params,
                            Tensor* output)
        {
            const auto& options = *static_cast<const ResizeParams*>(params);
            hpcplatform_RETURN_IF_ERROR(RejectNchw(options.layout));
            auto image = ParseImage(input.shape(), options.layout);
            if (!image.ok()) return image.status();
            ImageInfo output_info = image.value();
            // Output has the same rank/layout as input; only H/W differ.
            output_info.h = options.height;
            output_info.w = options.width;
            return CvCall(
                [&]
                {
                    for (int64_t n = 0; n < image.value().n; ++n)
                    {
                        // HxWxC view of image n; failure means dtype/contiguity violates OpenCV
                        // requirements.
                        auto source = ImageMat(input, image.value(), n);
                        auto destination = ImageMat(*output, output_info, n);
                        if (!source.ok() || !destination.ok())
                        {
                            throw cv::Exception(0, "invalid image tensor", __func__, __FILE__,
                                                __LINE__);
                        }
                        cv::resize(source.value(), destination.value(),
                                   cv::Size(static_cast<int>(options.width),
                                            static_cast<int>(options.height)),
                                   0, 0,
                                   // Map each interpolation mode to its OpenCV flag.
                                   options.interpolation == InterpolationMode::kNearest
                                       ? cv::INTER_NEAREST
                                       : cv::INTER_LINEAR);
                    }
                });
        }

    }  // namespace ops_internal
}  // namespace hpcplatform
