#include <cstring>
#include <iostream>

#include "mycv/mycv.h"

int main() {
  using namespace mycv;

  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto src = context->AllocateTensor(Shape{4}, DataType::kFloat32).value();
  auto dst = context->AllocateTensor(Shape{4}, DataType::kFloat32).value();

  for (int i = 0; i < 4; ++i) src.Data<float>()[i] = static_cast<float>(i + 1);

  Status status = context->memory().Copy(dst.buffer().get(), 0,
                                         src.buffer().get(), 0,
                                         src.ByteSize(), nullptr);
  if (!status.ok()) {
    std::cerr << "copy failed: " << status.ToString() << "\n";
    return 1;
  }

  const bool equal = std::memcmp(src.data(), dst.data(), src.ByteSize()) == 0;
  std::cout << "copy verified: " << (equal ? "yes" : "no") << "\n";

  const MemoryStats stats = context->memory().Stats();
  std::cout << "allocated=" << stats.allocated_bytes
            << " peak=" << stats.peak_bytes
            << " allocations=" << stats.allocation_count << "\n";
  return equal ? 0 : 1;
}
