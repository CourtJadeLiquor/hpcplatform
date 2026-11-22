// Demonstrates the CUDA backend: device-memory tensors, Host↔Device copies,
// and operator execution (safely skipped in non-CUDA builds).
#include <iostream>

#include "mycv/mycv.h"

int main() {
  using namespace mycv;
  auto cuda_result = Context::Create(BackendType::kCUDA);
  if (!cuda_result.ok()) {
    std::cout << "SKIP: " << cuda_result.status().ToString() << '\n';
    return 0;
  }
  auto cuda = cuda_result.value();
  auto native = Context::Create(BackendType::kNativeCpu).value();
  auto host = native->AllocateTensor(Shape{4, 4, 3}, DataType::kUInt8).value();
  for (int i = 0; i < 48; ++i)
    host.Data<uint8_t>()[i] = static_cast<uint8_t>(i * 5);
  auto device = cuda->AllocateTensor(host.shape(), host.dtype()).value();
  const Status upload = cuda->Copy(&device, host);
  if (!upload.ok()) return 1;

  OpContext ops;
  ops.context = cuda.get();
  ops.stream = &cuda->stream();
  auto resized = Resize(ops, device, 2, 2);
  if (!resized.ok()) return 1;
  auto blurred = GaussianBlur(ops, resized.value(), 3, 1.0);
  if (!blurred.ok()) return 1;
  auto output =
      native->AllocateTensor(blurred.value().shape(), blurred.value().dtype())
          .value();
  const Status download = native->Copy(&output, blurred.value());
  if (!download.ok()) return 1;

  std::cout << cuda->Describe() << '\n';
  std::cout << "CUDA pipeline: H2D -> resize -> gaussian_blur -> D2H\n";
  std::cout << output.ToString()
            << ", first=" << static_cast<int>(output.Data<uint8_t>()[0])
            << '\n';
  return 0;
}
