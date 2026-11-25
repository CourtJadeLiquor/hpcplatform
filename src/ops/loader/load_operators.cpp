// Public entry points for config-driven operator loading. This translation unit
// is always compiled: when hpcplatform_WITH_YAML is off it returns kUnsupported, so the
// public API stays linkable in every build. The real (yaml-cpp) implementation
// lives in load_operators_real.cpp and is only compiled with the feature on,
// keeping this file free of a yaml-cpp dependency.

#include <memory>
#include <string>

#include "hpcplatform/ops/operator.h"

namespace hpcplatform
{

#ifdef hpcplatform_WITH_YAML
    // Defined in load_operators_real.cpp. `shared_context` is nullptr for the
    // per-entry-environment overloads, or a pointer to the caller's context when
    // sharing is requested.
    Result<OperatorMap> LoadOperatorsFromFile(const std::string& config_path,
                                              const std::shared_ptr<Context>* shared_context);
    Result<OperatorMap> LoadOperatorsFromText(const std::string& yaml,
                                              const std::shared_ptr<Context>* shared_context);
#endif

    Result<OperatorMap> LoadOperators(const std::string& config_path)
    {
#ifdef hpcplatform_WITH_YAML
        return LoadOperatorsFromFile(config_path, /*shared_context=*/nullptr);
#else
        (void)config_path;
        return Status::Unsupported("LoadOperators requires hpcplatform_WITH_YAML");
#endif
    }

    Result<OperatorMap> LoadOperatorsFromString(const std::string& yaml)
    {
#ifdef hpcplatform_WITH_YAML
        return LoadOperatorsFromText(yaml, /*shared_context=*/nullptr);
#else
        (void)yaml;
        return Status::Unsupported("LoadOperators requires hpcplatform_WITH_YAML");
#endif
    }

    Result<OperatorMap> LoadOperators(const std::string& config_path,
                                      const std::shared_ptr<Context>& shared_context)
    {
#ifdef hpcplatform_WITH_YAML
        return LoadOperatorsFromFile(config_path, &shared_context);
#else
        (void)config_path;
        (void)shared_context;
        return Status::Unsupported("LoadOperators requires hpcplatform_WITH_YAML");
#endif
    }

}  // namespace hpcplatform
