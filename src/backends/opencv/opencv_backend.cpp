#include "backends/opencv/opencv_backend.h"

#include <opencv2/core.hpp>

#include "backends/backend_internal.h"

namespace hpcplatform {

Result<std::shared_ptr<Backend>> CreateOpenCvBackend(int device_index) {
  std::shared_ptr<Backend> backend =
      std::make_shared<OpenCvBackend>(device_index);
  return backend;
}

}  // namespace hpcplatform
