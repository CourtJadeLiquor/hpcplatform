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
  for (int i = 0; i < 48; ++i) host.Data<uint8_t>()[i] = static_cast<uint8_t>(i * 5);
  auto device = cuda->AllocateTensor(host.shape(), host.dtype()).value();
  const Status upload = cuda->Copy(&device, host);
  if (!upload.ok()) return 1;

  OpContext ops;
  ops.context = cuda.get();
  ops.stream = &cuda->stream();
  auto resized = Resize(ops, device, 2, 2);
  if (!resized.ok()) return 1;
  auto normalized = Normalize(ops, resized.value(), {0, 0, 0}, {255, 255, 255});
  if (!normalized.ok()) return 1;
  auto chw = Transpose(ops, normalized.value(), {2, 0, 1});
  if (!chw.ok()) return 1;
  auto output = native->AllocateTensor(chw.value().shape(), chw.value().dtype()).value();
  const Status download = native->Copy(&output, chw.value());
  if (!download.ok()) return 1;

  std::cout << cuda->backend().Describe() << '\n';
  std::cout << "CUDA pipeline: H2D -> resize -> normalize -> HWC-to-CHW -> D2H\n";
  std::cout << output.ToString() << ", first=" << output.Data<float>()[0] << '\n';
  return 0;
}
