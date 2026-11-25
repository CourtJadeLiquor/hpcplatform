// Demonstrates Context/Buffer allocation, Host↔Device copies, and
// Tensor::Clone lifetime behavior.
#include <cstring>
#include <iostream>

#include "hpcplatform/core/transfer.h"
#include "hpcplatform/hpcplatform.h"

int main()
{
    using namespace hpcplatform;

    auto context = Context::Create(BackendType::kNativeCpu).value();
    auto src = Tensor::Empty(*context, Shape{4}, DataType::kFloat32).value();
    auto dst = Tensor::Empty(*context, Shape{4}, DataType::kFloat32).value();

    for (int i = 0; i < 4; ++i) src.Data<float>()[i] = static_cast<float>(i + 1);

    const Status status = Copy(src, &dst);
    if (!status.ok())
    {
        std::cerr << "copy failed: " << status.ToString() << "\n";
        return 1;
    }

    const bool equal = std::memcmp(src.data(), dst.data(), src.ByteSize()) == 0;
    std::cout << "copy verified: " << (equal ? "yes" : "no") << "\n";

    const MemoryStats stats = context->GetMemoryStats();
    std::cout << "allocated=" << stats.allocated_bytes << " peak=" << stats.peak_bytes
              << " allocations=" << stats.allocation_count << "\n";
    return equal ? 0 : 1;
}
