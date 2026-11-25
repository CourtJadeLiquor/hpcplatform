// Config-driven concrete operator + descriptor for wavelet_transform. Compiled
// only when hpcplatform_WITH_YAML is on; the default build never references these
// symbols.

#include <memory>
#include <string>
#include <utility>

#include "framework/op_descriptor.h"
#include "hpcplatform/core/common.h"
#include "hpcplatform/core/context.h"
#include "hpcplatform/ops/image.h"
#include "hpcplatform/ops/op.h"
#include "hpcplatform/ops/wavelet.h"

namespace hpcplatform
{
    namespace
    {

        struct ParsedWaveletParams
        {
            WaveletKind kind;
            ImageLayout layout;
        };

        // Parses and validates the config params into typed values. Shared by
        // ValidateParams (load-time fail-fast) and Construct (object build).
        Result<ParsedWaveletParams> ParseWaveletParams(const ParamMap& params)
        {
            auto kind_name = ParamGetString(params, "kind");
            if (!kind_name.ok()) return kind_name.status();
            auto kind = WaveletKindFromName(kind_name.value());
            if (!kind.ok()) return kind.status();
            auto layout_name = ParamGetString(params, "layout");
            if (!layout_name.ok()) return layout_name.status();
            auto layout = ImageLayoutFromName(layout_name.value());
            if (!layout.ok()) return layout.status();
            return ParsedWaveletParams{kind.value(), layout.value()};
        }

        class WaveletTransformDescriptor final : public OpDescriptor
        {
        public:
            const char* Name() const override { return "wavelet_transform"; }
            OpKind Kind() const override { return OpKind::kWaveletTransform; }

            Status ValidateParams(const ParamMap& params) const override
            {
                auto parsed = ParseWaveletParams(params);
                return parsed.ok() ? Status::Ok() : parsed.status();
            }

            Result<std::unique_ptr<Operator>> Construct(const std::string& name,
                                                        const std::shared_ptr<Context>& context,
                                                        CpuKernelLevel cpu_level,
                                                        const ParamMap& params) const override
            {
                auto parsed = ParseWaveletParams(params);
                if (!parsed.ok()) return parsed.status();
                const ParsedWaveletParams& p = parsed.value();
                return std::unique_ptr<Operator>(std::make_unique<WaveletTransformOperator>(
                    name, context, cpu_level, p.kind, p.layout));
            }
        };

    }  // namespace

    void RegisterWaveletTransformDescriptor()
    {
        RegisterOpDescriptor(std::make_unique<WaveletTransformDescriptor>());
    }

    // --- WaveletTransformOperator ----------------------------------------------

    WaveletTransformOperator::WaveletTransformOperator(std::string name,
                                                       std::shared_ptr<Context> context,
                                                       CpuKernelLevel cpu_level,
                                                       WaveletKind kind,
                                                       ImageLayout layout)
        : name_(std::move(name)), kind_(kind), layout_(layout)
    {
        context_ = std::move(context);
        cpu_level_ = cpu_level;
    }

    const std::string& WaveletTransformOperator::name() const { return name_; }

    OpKind WaveletTransformOperator::kind() const { return OpKind::kWaveletTransform; }

    Result<WaveletBands> WaveletTransformOperator::Run(const Tensor& input) const
    {
        OpContext ctx(*context_);
        ctx.set_cpu_level(cpu_level_);
        return hpcplatform::WaveletTransform(ctx, input, kind_, layout_);
    }

}  // namespace hpcplatform
