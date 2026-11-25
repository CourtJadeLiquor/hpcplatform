#ifndef hpcplatform_OPS_OP_H_
#define hpcplatform_OPS_OP_H_

#include <cstdint>
#include <string>

#include "hpcplatform/core/common.h"
#include "hpcplatform/core/context.h"
#include "hpcplatform/core/stream.h"

namespace hpcplatform
{

    // Execution descriptor binding an operator invocation to a device context plus
    // scheduling hints. Constructors are restricted: production code should always
    // go through `OpContext::From(context)` or `OpContext(context)` — both paths
    // bind a non-null Context. The only way to obtain a null context is the test
    // helper `OpContext::Null()`.
    //
    // Deliberately an aggregate-free class so that callers cannot slip in a null
    // Context via brace initialization or forget to set stream/cpu_level.
    class OpContext
    {
    public:
        // No default construction in production code; every invocation must identify
        // a Context. The Null() factory below uses the private default construction
        // path only.

        // Binds the descriptor to `context` and its default stream. The CPU kernel
        // level defaults to kNaive, which is harmless on non-CPU backends.
        explicit OpContext(Context& context)
            : context_(&context), stream_(&context.stream()), cpu_level_(CpuKernelLevel::kNaive)
        {
        }

        // Same as the constructor; exists to match the project's factory-style API.
        static OpContext From(Context& context) { return OpContext(context); }

        // Returns an OpContext with no bound Context. Use only in error-path tests
        // where an operator must detect a null owning Context.
        static OpContext Null() { return OpContext(kNullTag{}); }

        // Fluent setter for advanced callers (benchmarking, CPU-backend dispatch).
        OpContext& set_stream(Stream* stream)
        {
            stream_ = stream;
            return *this;
        }
        OpContext& set_cpu_level(CpuKernelLevel level)
        {
            cpu_level_ = level;
            return *this;
        }

        Context* context() const { return context_; }
        Stream* stream() const { return stream_; }
        CpuKernelLevel cpu_level() const { return cpu_level_; }

    private:
        struct kNullTag
        {
        };
        // Private path exclusively used by Null(); no public caller can reach this.
        explicit OpContext(kNullTag)
            : context_(nullptr), stream_(nullptr), cpu_level_(CpuKernelLevel::kNaive)
        {
        }

        Context* context_ = nullptr;
        Stream* stream_ = nullptr;
        CpuKernelLevel cpu_level_ = CpuKernelLevel::kNaive;
    };

    // Stable operator IDs: append only, never reorder; numeric values remain stable
    // across versions.
    enum class OpKind : uint16_t
    {
        kResize = 1,
        kGaussianBlur = 2,
        kLaplacianPyramid = 3,
        kFourierTransform = 4,
        kWaveletTransform = 5,
    };

    // Returns the operator's stable short name for logs, profiling labels, and errors.
    inline const char* OpKindName(OpKind kind)
    {
        switch (kind)
        {
            case OpKind::kResize:
                return "resize";
            case OpKind::kGaussianBlur:
                return "gaussian_blur";
            case OpKind::kLaplacianPyramid:
                return "laplacian_pyramid";
            case OpKind::kFourierTransform:
                return "fourier_transform";
            case OpKind::kWaveletTransform:
                return "wavelet_transform";
        }
        return "unknown";
    }

    // Parses a stable operator name back to its enum. Inverse of OpKindName
    // (excluding "unknown"); unknown name -> kInvalidArgument.
    inline Result<OpKind> OpKindFromName(const std::string& name)
    {
        if (name == "resize") return OpKind::kResize;
        if (name == "gaussian_blur") return OpKind::kGaussianBlur;
        if (name == "laplacian_pyramid") return OpKind::kLaplacianPyramid;
        if (name == "fourier_transform") return OpKind::kFourierTransform;
        if (name == "wavelet_transform") return OpKind::kWaveletTransform;
        return Status::InvalidArgument("unknown operator name: " + name);
    }

}  // namespace hpcplatform

#endif  // hpcplatform_OPS_OP_H_
