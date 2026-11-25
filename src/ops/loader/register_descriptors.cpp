// Explicit manifest that registers every operator descriptor compiled into the
// binary. Mirrors the kernel manifest pattern (src/ops/register_ops.cpp):
// static self-registration is unreliable in a static library, so registration
// is driven from this one referenced translation unit. Compiled only when
// hpcplatform_WITH_YAML is on (see top-level CMakeLists.txt).

#include "framework/op_descriptor.h"

namespace hpcplatform
{

    // Each operator owns its descriptor registration in its own op folder. Add one
    // line here to register a new config-driven operator; no shared loader code
    // needs to change (open/closed, AC-7).
    void RegisterResizeDescriptor();
    void RegisterGaussianBlurDescriptor();
    void RegisterLaplacianPyramidDescriptor();
    void RegisterFourierTransformDescriptor();
    void RegisterWaveletTransformDescriptor();

    void RegisterAllOpDescriptors()
    {
        RegisterResizeDescriptor();
        RegisterGaussianBlurDescriptor();
        RegisterLaplacianPyramidDescriptor();
        RegisterFourierTransformDescriptor();
        RegisterWaveletTransformDescriptor();
    }

}  // namespace hpcplatform
