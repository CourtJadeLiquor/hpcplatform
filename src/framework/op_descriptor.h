#ifndef hpcplatform_FRAMEWORK_OP_DESCRIPTOR_H_
#define hpcplatform_FRAMEWORK_OP_DESCRIPTOR_H_

#include <map>
#include <memory>
#include <string>

#include "hpcplatform/core/common.h"
#include "hpcplatform/core/context.h"
#include "hpcplatform/core/error.h"
#include "hpcplatform/ops/op.h"
#include "hpcplatform/ops/operator.h"

namespace hpcplatform
{

    // String-valued operator parameters parsed from a config. Scalars are carried
    // as strings and converted to typed values by each descriptor, keeping the
    // framework config loader free of per-op parameter knowledge.
    using ParamMap = std::map<std::string, std::string>;

    // Op-agnostic descriptor registered by each operator. The config loader resolves
    // a config entry's `op` name to its descriptor, asks it to validate parameters,
    // checks kernel availability via FindKernel (using Kind()), then asks it to
    // construct a configured Operator. Decouples the loader from concrete operators
    // so adding an operator does not touch shared loader code (open/closed).
    class OpDescriptor
    {
    public:
        virtual ~OpDescriptor() = default;

        // Stable name used as the config `op` value (e.g. "resize").
        virtual const char* Name() const = 0;
        // OpKind used for kernel availability lookup via FindKernel.
        virtual OpKind Kind() const = 0;
        // Validates the parameter map without constructing anything; missing required
        // values or wrong types -> kInvalidArgument.
        virtual Status ValidateParams(const ParamMap& params) const = 0;
        // Constructs a configured Operator bound to the given context/cpu_level and
        // the parsed params. `name` is the instance name from the config (the key in
        // the returned map). Caller must have run ValidateParams and confirmed kernel
        // availability already.
        virtual Result<std::unique_ptr<Operator>> Construct(const std::string& name,
                                                            const std::shared_ptr<Context>& context,
                                                            CpuKernelLevel cpu_level,
                                                            const ParamMap& params) const = 0;
    };

    // Required-value accessors shared by descriptors. A missing key or a value that
    // cannot be parsed to the target type yields kInvalidArgument. Implemented with
    // strtoll/strtod (no throwing converters) so the parsers stay exception-free
    // even though the library does not throw (see hpcplatform/core/error.h).
    Result<int64_t> ParamGetInt64(const ParamMap& params, const std::string& key);
    Result<double> ParamGetDouble(const ParamMap& params, const std::string& key);
    Result<std::string> ParamGetString(const ParamMap& params, const std::string& key);

    // Registers a descriptor (takes ownership). Safe to call before the registry is
    // initialized. Registration is explicit (an ops-internal manifest calls this),
    // never static self-registration, matching op_registry's linking guarantees.
    void RegisterOpDescriptor(std::unique_ptr<OpDescriptor> descriptor);

    // Looks up a descriptor by name; returns nullptr if not registered. Triggers
    // one-time registration of all descriptors on first use.
    const OpDescriptor* FindOpDescriptor(const std::string& name);

    // Registers every operator descriptor compiled into this binary. Declared here,
    // defined in the operator module manifest (src/ops/loader/register_descriptors.cpp).
    // Only referenced when hpcplatform_WITH_YAML is enabled, so builds without the feature
    // never pull in operator descriptors.
    void RegisterAllOpDescriptors();

}  // namespace hpcplatform

#endif  // hpcplatform_FRAMEWORK_OP_DESCRIPTOR_H_
