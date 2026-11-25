// Every data-movement entry point on one page, with the rule for choosing:
//
//   Copy(src, &dst)                 direction-agnostic primitive
//   Upload(host_src, &device_dst)   H2D intent + memory-kind asserts
//   Download(device_src, &host_dst) D2H intent + memory-kind asserts
//   Tensor::From / Tensor::Into     typed adapters; they own the staging
//
// All three transfer functions take (source, destination), are synchronous, and
// infer the executing backend from tensor ownership.
#include <cstdint>
#include <iostream>

#ifdef hpcplatform_WITH_OPENCV
#include <opencv2/core.hpp>
#endif

#include "hpcplatform/hpcplatform.h"

namespace
{

    int g_failures = 0;

    void Check(const hpcplatform::Status& status, const char* step)
    {
        if (status.ok())
        {
            std::cout << "  [ ok ] " << step << '\n';
            return;
        }
        ++g_failures;
        std::cout << "  [FAIL] " << step << ": " << status.ToString() << '\n';
    }

    void Section(const char* title) { std::cout << "\n== " << title << " ==\n"; }

}  // namespace

int main()
{
    using namespace hpcplatform;

    auto cpu_result = Context::Create(BackendType::kNativeCpu);
    if (!cpu_result.ok())
    {
        std::cerr << cpu_result.status().ToString() << '\n';
        return 1;
    }
    auto cpu = cpu_result.value();

    auto source = Tensor::Empty(*cpu, Shape{2, 3}, DataType::kInt32).value();
    for (int i = 0; i < 6; ++i) source.Data<int32_t>()[i] = i + 1;

    // 1. host -> host -------------------------------------------------------
    // No direction to assert: Copy is the primitive.
    Section("host -> host: Copy");
    auto host_dst = Tensor::Empty(*cpu, source.shape(), source.dtype()).value();
    Check(Copy(source, &host_dst), "Copy(source, &host_dst)");
    std::cout << "  values = " << host_dst.Data<int32_t>()[0] << ',' << host_dst.Data<int32_t>()[3]
              << ',' << host_dst.Data<int32_t>()[5] << '\n';

    // 2. non-contiguous view -----------------------------------------------
    // Copy walks logical elements, so a strided view lands densely.
    Section("strided view -> contiguous: Copy");
    auto view = source.Slice(1, 0, 2).value();  // 2x2 out of a 2x3 tensor
    auto view_dst = Tensor::Empty(*cpu, view.shape(), view.dtype()).value();
    Check(Copy(view, &view_dst), "Copy(view, &contiguous_dst)");
    std::cout << "  source contiguous? " << (view.IsContiguous() ? "yes" : "no")
              << "  result = " << view_dst.ToString() << " [" << view_dst.Data<int32_t>()[0] << ','
              << view_dst.Data<int32_t>()[1] << ',' << view_dst.Data<int32_t>()[2] << ','
              << view_dst.Data<int32_t>()[3] << "]\n";

    // 3. host <-> device ---------------------------------------------------
    // The direction is fixed by the code that follows, so name it.
    auto cuda_result = Context::Create(BackendType::kCUDA);
    if (cuda_result.ok())
    {
        auto cuda = cuda_result.value();

        Section("host -> device: Upload, device -> host: Download");
        auto device = Tensor::Empty(*cuda, source.shape(), source.dtype()).value();
        Check(Upload(source, &device), "Upload(host_src, &device_dst)");
        auto back = Tensor::Empty(*cpu, source.shape(), source.dtype()).value();
        Check(Download(device, &back), "Download(device_src, &host_dst)");
        std::cout << "  round-trip intact? " << (back.Data<int32_t>()[5] == 6 ? "yes" : "no")
                  << '\n';

        Section("device -> device: Copy / Clone");
        auto device_dst = Tensor::Empty(*cuda, source.shape(), source.dtype()).value();
        Check(Copy(device, &device_dst), "Copy(device_src, &device_dst)");
        auto cloned = device.Clone();
        Check(cloned.status(), "device.Clone()");
    }
    else
    {
        std::cout << "\n== device sections skipped: " << cuda_result.status().ToString() << " ==\n";
    }

    // 4. third-party type --------------------------------------------------
    // The adapter owns the staging: user code never spells Upload/Download.
#ifdef hpcplatform_WITH_OPENCV
    Section("cv::Mat <-> Tensor: From / Into");
    cv::Mat mat(2, 2, CV_8UC1);
    for (int i = 0; i < 4; ++i) mat.data[i] = static_cast<unsigned char>(i * 10);
    auto from_mat = Tensor::From(mat, *cpu);
    Check(from_mat.status(), "Tensor::From(mat, cpu_context)");
    auto to_mat = Tensor::Into<cv::Mat>(from_mat.value());
    Check(to_mat.status(), "Tensor::Into<cv::Mat>(tensor)");
    std::cout << "  mat -> tensor -> mat, first = "
              << static_cast<int>(to_mat.value().at<unsigned char>(0, 0)) << '\n';
#else
    std::cout << "\n== cv::Mat section skipped: built without OpenCV ==\n";
#endif

    std::cout << "\nsummary: " << (g_failures == 0 ? "all steps passed" : "FAILED") << '\n';
    return g_failures == 0 ? 0 : 1;
}
