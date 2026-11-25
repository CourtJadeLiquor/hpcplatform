#include "hpcplatform/core/common.h"

#include <sstream>

namespace hpcplatform {

std::string Shape::ToString() const {
  if (rank_ == 0) return "scalar";
  std::ostringstream oss;
  for (int i = 0; i < rank_; ++i) {
    if (i > 0) oss << 'x';
    oss << dims_[static_cast<size_t>(i)];
  }
  return oss.str();
}

}  // namespace hpcplatform
