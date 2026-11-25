// Demonstrates the CUDA backend: device-memory tensors, Host↔Device copies,
// and operator execution (safely skipped in non-CUDA builds).
#include <iostream>

#include "hpcplatform/core/transfer.h"
#include "hpcplatform/hpcplatform.h"

int main()
{
    using namespace hpcplatform;
    auto cuda_result = Context::Create(BackendType::kCUDA);
    if (!cuda_result.ok())
    {
        std::cout << "SKIP: " << cuda_result.status().ToString() << '\n';
        return 0;
    }
    auto cuda = cuda_result.value();
    auto native = Context::Create(BackendType::kNativeCpu).value();
    auto host = Tensor::Empty(*native, Shape{4, 4, 3}, DataType::kUInt8).value();
    for (int i = 0; i < 48; ++i) host.Data<uint8_t>()[i] = static_cast<uint8_t>(i * 5);
    auto device = Tensor::Empty(*cuda, host.shape(), host.dtype()).value();
    // The direction is known here, so name it: Upload asserts that the source is
    // host memory and the destination is device memory.
    const Status upload = Upload(host, &device);
    if (!upload.ok()) return 1;

    OpContext ops(*cuda.get());
    auto resized = Resize(ops, device, 2, 2);
    if (!resized.ok()) return 1;
    auto blurred = GaussianBlur(ops, resized.value(), 3, 1.0);
    if (!blurred.ok()) return 1;
    auto output = Tensor::Empty(*native, blurred.value().shape(), blurred.value().dtype()).value();
    const Status download = Download(blurred.value(), &output);
    if (!download.ok()) return 1;

    std::cout << cuda->Describe() << '\n';
    std::cout << "CUDA pipeline: H2D -> resize -> gaussian_blur -> D2H\n";
    std::cout << output.ToString() << ", first=" << static_cast<int>(output.Data<uint8_t>()[0])
              << '\n';
    return 0;
}
