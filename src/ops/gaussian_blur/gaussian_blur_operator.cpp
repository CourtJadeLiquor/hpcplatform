// Config-driven concrete operator + descriptor for gaussian_blur. Compiled only
// when MYCV_WITH_YAML is on; the default build never references these symbols.

#include "mycv/ops/gaussian_blur.h"

#include <climits>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

#include "framework/op_descriptor.h"
#include "mycv/core/context.h"
#include "mycv/core/kernel_level.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"

namespace mycv {
namespace {

struct ParsedGaussianBlurParams {
  int kernel_size;
  double sigma;
  ImageLayout layout;
};

// Parses and validates the config params into typed values. Shared by
// ValidateParams (load-time fail-fast) and Construct (object build).
Result<ParsedGaussianBlurParams> ParseGaussianBlurParams(
    const ParamMap& params) {
  auto kernel_size = ParamGetInt64(params, "kernel_size");
  if (!kernel_size.ok()) return kernel_size.status();
  if (kernel_size.value() < 3 || (kernel_size.value() % 2) == 0 ||
      kernel_size.value() > INT_MAX) {
    return Status::InvalidArgument("kernel_size must be an odd integer >= 3");
  }
  auto sigma = ParamGetDouble(params, "sigma");
  if (!sigma.ok()) return sigma.status();
  if (!(sigma.value() > 0.0) || !std::isfinite(sigma.value())) {
    return Status::InvalidArgument("sigma must be finite and > 0");
  }
  auto layout_name = ParamGetString(params, "layout");
  if (!layout_name.ok()) return layout_name.status();
  auto layout = ImageLayoutFromName(layout_name.value());
  if (!layout.ok()) return layout.status();
  return ParsedGaussianBlurParams{static_cast<int>(kernel_size.value()),
                                  sigma.value(), layout.value()};
}

class GaussianBlurDescriptor final : public OpDescriptor {
 public:
  const char* Name() const override { return "gaussian_blur"; }
  OpKind Kind() const override { return OpKind::kGaussianBlur; }

  Status ValidateParams(const ParamMap& params) const override {
    auto parsed = ParseGaussianBlurParams(params);
    return parsed.ok() ? Status::Ok() : parsed.status();
  }

  Result<std::unique_ptr<Operator>> Construct(
      const std::string& name, const std::shared_ptr<Context>& context,
      CpuKernelLevel cpu_level, const ParamMap& params) const override {
    auto parsed = ParseGaussianBlurParams(params);
    if (!parsed.ok()) return parsed.status();
    const ParsedGaussianBlurParams& p = parsed.value();
    return std::unique_ptr<Operator>(std::make_unique<GaussianBlurOperator>(
        name, context, cpu_level, p.kernel_size, p.sigma, p.layout));
  }
};

}  // namespace

void RegisterGaussianBlurDescriptor() {
  RegisterOpDescriptor(std::make_unique<GaussianBlurDescriptor>());
}

// --- GaussianBlurOperator ---------------------------------------------------

GaussianBlurOperator::GaussianBlurOperator(std::string name,
                                           std::shared_ptr<Context> context,
                                           CpuKernelLevel cpu_level,
                                           int kernel_size, double sigma,
                                           ImageLayout layout)
    : name_(std::move(name)),
      kernel_size_(kernel_size),
      sigma_(sigma),
      layout_(layout) {
  context_ = std::move(context);
  cpu_level_ = cpu_level;
}

const std::string& GaussianBlurOperator::name() const { return name_; }

OpKind GaussianBlurOperator::kind() const { return OpKind::kGaussianBlur; }

Result<Tensor> GaussianBlurOperator::Run(const Tensor& input) const {
  const OpContext ctx{context_.get(), /*stream=*/nullptr, cpu_level_};
  return mycv::GaussianBlur(ctx, input, kernel_size_, sigma_, layout_);
}

}  // namespace mycv
