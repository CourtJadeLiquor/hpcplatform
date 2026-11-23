// Config-driven concrete operator + descriptor for fourier_transform. Compiled
// only when hpcplatform_WITH_YAML is on; the default build never references these
// symbols.

#include "hpcplatform/ops/fourier.h"

#include <memory>
#include <string>
#include <utility>

#include "framework/op_descriptor.h"
#include "hpcplatform/core/context.h"
#include "hpcplatform/core/kernel_level.h"
#include "hpcplatform/ops/image.h"
#include "hpcplatform/ops/op.h"

namespace hpcplatform {
namespace {

struct ParsedFourierParams {
  ImageLayout layout;
};

// Parses and validates the config params into typed values. Shared by
// ValidateParams (load-time fail-fast) and Construct (object build).
Result<ParsedFourierParams> ParseFourierParams(const ParamMap& params) {
  auto layout_name = ParamGetString(params, "layout");
  if (!layout_name.ok()) return layout_name.status();
  auto layout = ImageLayoutFromName(layout_name.value());
  if (!layout.ok()) return layout.status();
  return ParsedFourierParams{layout.value()};
}

class FourierTransformDescriptor final : public OpDescriptor {
 public:
  const char* Name() const override { return "fourier_transform"; }
  OpKind Kind() const override { return OpKind::kFourierTransform; }

  Status ValidateParams(const ParamMap& params) const override {
    auto parsed = ParseFourierParams(params);
    return parsed.ok() ? Status::Ok() : parsed.status();
  }

  Result<std::unique_ptr<Operator>> Construct(
      const std::string& name, const std::shared_ptr<Context>& context,
      CpuKernelLevel cpu_level, const ParamMap& params) const override {
    auto parsed = ParseFourierParams(params);
    if (!parsed.ok()) return parsed.status();
    return std::unique_ptr<Operator>(std::make_unique<FourierTransformOperator>(
        name, context, cpu_level, parsed.value().layout));
  }
};

}  // namespace

void RegisterFourierTransformDescriptor() {
  RegisterOpDescriptor(std::make_unique<FourierTransformDescriptor>());
}

// --- FourierTransformOperator ----------------------------------------------

FourierTransformOperator::FourierTransformOperator(
    std::string name, std::shared_ptr<Context> context,
    CpuKernelLevel cpu_level, ImageLayout layout)
    : name_(std::move(name)), layout_(layout) {
  context_ = std::move(context);
  cpu_level_ = cpu_level;
}

const std::string& FourierTransformOperator::name() const { return name_; }

OpKind FourierTransformOperator::kind() const {
  return OpKind::kFourierTransform;
}

Result<ComplexImage> FourierTransformOperator::Run(const Tensor& input) const {
  const OpContext ctx{context_.get(), /*stream=*/nullptr, cpu_level_};
  return hpcplatform::FourierTransform(ctx, input, layout_);
}

}  // namespace hpcplatform
