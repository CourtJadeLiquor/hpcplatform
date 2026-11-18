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
  auto normalized = Normalize(op_context, resized.value(), {0, 0, 0}, {255, 255, 255});
  if (!normalized.ok()) return 1;
  auto chw = Transpose(op_context, normalized.value(), {2, 0, 1});
  if (!chw.ok()) return 1;
  auto half = Cast(op_context, chw.value(), DataType::kFloat16);
  if (!half.ok()) return 1;
  auto restored = Cast(op_context, half.value(), DataType::kFloat32);
  if (!restored.ok()) return 1;

  std::cout << "pipeline: HWC u8 -> resize -> normalize -> CHW -> f16 -> f32\n";
  std::cout << restored.value().ToString() << '\n' << std::fixed << std::setprecision(4);
  for (int64_t i = 0; i < restored.value().Numel(); ++i) {
    std::cout << restored.value().Data<float>()[i] << (i + 1 == restored.value().Numel() ? '\n' : ' ');
  }
  return 0;
}
