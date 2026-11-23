// Config-driven concrete operator + descriptor for resize. Compiled only when
// hpcplatform_WITH_YAML is on; the default build never references these symbols.

#include "hpcplatform/ops/resize.h"

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

struct ParsedResizeParams {
  int64_t output_height;
  int64_t output_width;
  InterpolationMode interpolation;
  ImageLayout layout;
};

// Parses and validates the config params into typed values. Shared by
// ValidateParams (load-time fail-fast) and Construct (object build), so the two
// paths cannot drift apart.
Result<ParsedResizeParams> ParseResizeParams(const ParamMap& params) {
  auto height = ParamGetInt64(params, "output_height");
  if (!height.ok()) return height.status();
  auto width = ParamGetInt64(params, "output_width");
  if (!width.ok()) return width.status();
  if (height.value() <= 0) {
    return Status::InvalidArgument("output_height must be > 0");
  }
  if (width.value() <= 0) {
    return Status::InvalidArgument("output_width must be > 0");
  }
  auto interp_name = ParamGetString(params, "interpolation");
  if (!interp_name.ok()) return interp_name.status();
  auto interpolation = InterpolationModeFromName(interp_name.value());
  if (!interpolation.ok()) return interpolation.status();
  auto layout_name = ParamGetString(params, "layout");
  if (!layout_name.ok()) return layout_name.status();
  auto layout = ImageLayoutFromName(layout_name.value());
  if (!layout.ok()) return layout.status();
  return ParsedResizeParams{height.value(), width.value(), interpolation.value(),
                            layout.value()};
}

class ResizeDescriptor final : public OpDescriptor {
 public:
  const char* Name() const override { return "resize"; }
  OpKind Kind() const override { return OpKind::kResize; }

  Status ValidateParams(const ParamMap& params) const override {
    auto parsed = ParseResizeParams(params);
    return parsed.ok() ? Status::Ok() : parsed.status();
  }

  Result<std::unique_ptr<Operator>> Construct(
      const std::string& name, const std::shared_ptr<Context>& context,
      CpuKernelLevel cpu_level, const ParamMap& params) const override {
    auto parsed = ParseResizeParams(params);
    if (!parsed.ok()) return parsed.status();
    const ParsedResizeParams& p = parsed.value();
    return std::unique_ptr<Operator>(std::make_unique<ResizeOperator>(
        name, context, cpu_level, p.output_height, p.output_width,
        p.interpolation, p.layout));
  }
};

}  // namespace

void RegisterResizeDescriptor() {
  RegisterOpDescriptor(std::make_unique<ResizeDescriptor>());
}

// --- ResizeOperator ---------------------------------------------------------

ResizeOperator::ResizeOperator(std::string name,
                               std::shared_ptr<Context> context,
                               CpuKernelLevel cpu_level, int64_t output_height,
                               int64_t output_width,
                               InterpolationMode interpolation,
                               ImageLayout layout)
    : name_(std::move(name)),
      output_height_(output_height),
      output_width_(output_width),
      interpolation_(interpolation),
      layout_(layout) {
  context_ = std::move(context);
  cpu_level_ = cpu_level;
}

const std::string& ResizeOperator::name() const { return name_; }

OpKind ResizeOperator::kind() const { return OpKind::kResize; }

Result<Tensor> ResizeOperator::Run(const Tensor& input) const {
  const OpContext ctx{context_.get(), /*stream=*/nullptr, cpu_level_};
  return hpcplatform::Resize(ctx, input, output_height_, output_width_,
                      interpolation_, layout_);
}

}  // namespace hpcplatform
