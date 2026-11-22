// Minimal public API example demonstrating Tensor creation, filling, views, and printing.
#include <iostream>

#include "mycv/mycv.h"

int main() {
  using namespace mycv;

  auto context_result = Context::Create(BackendType::kNativeCpu);
  if (!context_result.ok()) {
    std::cerr << "failed to create context: "
              << context_result.status().ToString() << "\n";
    return 1;
  }
  auto context = context_result.value();
  std::cout << "mycv " << MYCV_VERSION_STRING << " | " << context->Describe()
            << "\n";

  auto tensor_result = context->AllocateTensor(Shape{2, 3}, DataType::kFloat32);
  if (!tensor_result.ok()) {
    std::cerr << "failed to allocate tensor: "
              << tensor_result.status().ToString() << "\n";
    return 1;
  }
  Tensor tensor = tensor_result.value();
  std::cout << tensor.ToString() << "\n";
  std::cout << "strides: [" << tensor.Stride(0) << ", " << tensor.Stride(1)
            << "], alignment: " << 64 << " bytes\n";

  float* data = tensor.Data<float>();
  for (int i = 0; i < 6; ++i)
    data[i] = static_cast<float>(i) * 0.5f;

  for (int row = 0; row < 2; ++row) {
    for (int col = 0; col < 3; ++col) {
      std::cout << data[row * 3 + col] << (col == 2 ? '\n' : ' ');
    }
  }
  return 0;
}
