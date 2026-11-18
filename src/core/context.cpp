#include "mycv/core/context.h"

#include <utility>

namespace mycv {

Result<std::shared_ptr<Context>> Context::Create(BackendType backend,
                                                 int device_index) {
  auto backend_result = CreateBackend(backend, device_index);
  if (!backend_result.ok()) return backend_result.status();
  return std::make_shared<Context>(std::move(backend_result.value()));
}

}  // namespace mycv
