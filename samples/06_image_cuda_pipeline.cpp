// End-to-end image pipeline across runtimes. Tensor::From stages a cv::Mat into
// device memory and Tensor::Into downloads it back, so the algorithm code names
// only the CUDA context: gaussian_blur + resize run on the GPU with no explicit
// H2D/D2H calls.
#include <algorithm>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "hpcplatform/hpcplatform.h"

namespace
{

    constexpr char kDefaultInputPath[] = hpcplatform_SOURCE_DIR "/resources/images/IMT.png";

}  // namespace

int main(int argc, char** argv)
{
    using namespace hpcplatform;
    const std::string input_path = argc > 1 ? argv[1] : kDefaultInputPath;
    const std::string output_path = argc > 2 ? argv[2] : "cuda_image_pipeline.png";

    const cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "cannot read image: " << input_path << '\n';
        return 1;
    }

    auto cuda_result = Context::Create(BackendType::kCUDA);
    if (!cuda_result.ok())
    {
        std::cout << "SKIP: " << cuda_result.status().ToString() << '\n';
        return 0;
    }
    const auto cuda = cuda_result.value();

    // Tensor::From owns the host -> device staging, so the algorithm only names
    // the Context that will execute it.
    auto device = Tensor::From(image, *cuda);
    if (!device.ok())
    {
        std::cerr << device.status().ToString() << '\n';
        return 1;
    }

    OpContext ops = OpContext::From(*cuda);
    auto blurred = GaussianBlur(ops, device.value(), 5, 1.5);
    if (!blurred.ok())
    {
        std::cerr << blurred.status().ToString() << '\n';
        return 1;
    }
    const int64_t output_height = std::max<int64_t>(1, image.rows / 2);
    const int64_t output_width = std::max<int64_t>(1, image.cols / 2);
    auto resized = Resize(ops, blurred.value(), output_height, output_width);
    if (!resized.ok())
    {
        std::cerr << resized.status().ToString() << '\n';
        return 1;
    }

    // Tensor::Into downloads from the device and returns an owning cv::Mat, so
    // no explicit D2H step (and no CPU Context) is needed.
    auto output = Tensor::Into<cv::Mat>(resized.value());
    if (!output.ok())
    {
        std::cerr << output.status().ToString() << '\n';
        return 1;
    }

    if (!cv::imwrite(output_path, output.value()))
    {
        std::cerr << "cannot write image: " << output_path << '\n';
        return 1;
    }

    std::cout << cuda->Describe() << '\n'
              << "CUDA image pipeline: " << image.cols << 'x' << image.rows << " -> "
              << output.value().cols << 'x' << output.value().rows << " -> " << output_path << '\n';
    return 0;
}
