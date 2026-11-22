// Config-driven concrete operator + descriptor for laplacian_pyramid. Compiled
// only when MYCV_WITH_YAML is on; the default build never references these
// symbols.

#include "mycv/ops/laplacian_pyramid.h"

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

struct ParsedLaplacianParams {
  int levels;
  int kernel_size;
  double sigma;
  ImageLayout layout;
};

// Parses and validates the config params into typed values. Shared by
// ValidateParams (load-time fail-fast) and Construct (object build).
Result<ParsedLaplacianParams> ParseLaplacianParams(const ParamMap& params) {
  auto levels = ParamGetInt64(params, "levels");
  if (!levels.ok()) return levels.status();
  if (levels.value() < 1 || levels.value() > INT_MAX) {
    return Status::InvalidArgument("levels must be an integer >= 1");
  }
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
  return ParsedLaplacianParams{static_cast<int>(levels.value()),
                               static_cast<int>(kernel_size.value()),
                               sigma.value(), layout.value()};
}

class LaplacianPyramidDescriptor final : public OpDescriptor {
 public:
  const char* Name() const override { return "laplacian_pyramid"; }
  OpKind Kind() const override { return OpKind::kLaplacianPyramid; }

  Status ValidateParams(const ParamMap& params) const override {
    auto parsed = ParseLaplacianParams(params);
    return parsed.ok() ? Status::Ok() : parsed.status();
  }

  Result<std::unique_ptr<Operator>> Construct(
      const std::string& name, const std::shared_ptr<Context>& context,
      CpuKernelLevel cpu_level, const ParamMap& params) const override {
    auto parsed = ParseLaplacianParams(params);
    if (!parsed.ok()) return parsed.status();
    const ParsedLaplacianParams& p = parsed.value();
    return std::unique_ptr<Operator>(std::make_unique<LaplacianPyramidOperator>(
        name, context, cpu_level, p.levels, p.kernel_size, p.sigma, p.layout));
  }
};

}  // namespace

void RegisterLaplacianPyramidDescriptor() {
  RegisterOpDescriptor(std::make_unique<LaplacianPyramidDescriptor>());
}

// --- LaplacianPyramidOperator ----------------------------------------------

LaplacianPyramidOperator::LaplacianPyramidOperator(
    std::string name, std::shared_ptr<Context> context,
    CpuKernelLevel cpu_level, int levels, int kernel_size, double sigma,
    ImageLayout layout)
    : name_(std::move(name)),
      levels_(levels),
      kernel_size_(kernel_size),
      sigma_(sigma),
      layout_(layout) {
  context_ = std::move(context);
  cpu_level_ = cpu_level;
}

const std::string& LaplacianPyramidOperator::name() const { return name_; }

OpKind LaplacianPyramidOperator::kind() const {
  return OpKind::kLaplacianPyramid;
}

Result<LaplacianPyramid> LaplacianPyramidOperator::Run(
    const Tensor& input) const {
  const OpContext ctx{context_.get(), /*stream=*/nullptr, cpu_level_};
  return mycv::BuildLaplacianPyramid(ctx, input, levels_, kernel_size_, sigma_,
                                     layout_);
}

}  // namespace mycv
