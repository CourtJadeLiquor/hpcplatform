#include <iomanip>
#include <iostream>

#include "mycv/mycv.h"

int main() {
  using namespace mycv;
  auto context_result = Context::Create(BackendType::kNativeCpu);
  if (!context_result.ok()) {
    std::cerr << context_result.status().ToString() << '\n';
    return 1;
  }
  auto context = context_result.value();
  auto input = context->AllocateTensor(Shape{4, 4, 3}, DataType::kUInt8).value();
  for (int i = 0; i < 48; ++i) input.Data<uint8_t>()[i] = static_cast<uint8_t>(i * 5);

  OpContext op_context;
  op_context.context = context.get();
  op_context.stream = &context->stream();
  op_context.cpu_level = CpuKernelLevel::kNaive;

  auto resized = Resize(op_context, input, 2, 2);
  if (!resized.ok()) return 1;
  auto blurred = GaussianBlur(op_context, resized.value(), 3, 1.0);
  if (!blurred.ok()) return 1;

  std::cout << "pipeline: HWC u8 -> resize -> gaussian_blur\n";
  std::cout << blurred.value().ToString() << '\n' << std::fixed << std::setprecision(2);
  for (int64_t i = 0; i < blurred.value().Numel(); ++i) {
    std::cout << static_cast<int>(blurred.value().Data<uint8_t>()[i])
              << (i + 1 == blurred.value().Numel() ? '\n' : ' ');
  }
  return 0;
}
