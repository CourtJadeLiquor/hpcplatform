#include "mycv/backends/opencv.h"

#include <cstring>
#include <limits>

namespace mycv {
namespace {

Result<DataType> DepthToDType(int depth) {
  switch (depth) {
    case CV_8U: return DataType::kUInt8;
    case CV_32F: return DataType::kFloat32;
    case CV_64F: return DataType::kFloat64;
    default:
      return Status::Unsupported("cv::Mat conversion supports u8, f32 and f64 only");
  }
}

int DTypeToDepth(DataType dtype) {
  switch (dtype) {
    case DataType::kUInt8: return CV_8U;
    case DataType::kFloat32: return CV_32F;
    case DataType::kFloat64: return CV_64F;
    default: return -1;
  }
}

}  // namespace

Result<Tensor> CopyFromCvMat(Context& context, const cv::Mat& mat) {
  if (context.device().type != DeviceType::kCPU) {
    return Status::Unsupported("CopyFromCvMat currently requires a CPU context");
  }
  if (mat.empty() || mat.dims != 2) {
    return Status::InvalidArgument("CopyFromCvMat requires a non-empty 2D cv::Mat");
  }
  auto dtype = DepthToDType(mat.depth());
  if (!dtype.ok()) return dtype.status();

  const Shape shape = mat.channels() == 1
                          ? Shape{mat.rows, mat.cols}
                          : Shape{mat.rows, mat.cols, mat.channels()};
  auto tensor = context.AllocateTensor(shape, dtype.value());
  if (!tensor.ok()) return tensor.status();

  if (mat.isContinuous()) {
    std::memcpy(tensor.value().data(), mat.data, tensor.value().ByteSize());
  } else {
    cv::Mat contiguous = mat.clone();
    std::memcpy(tensor.value().data(), contiguous.data, tensor.value().ByteSize());
  }
  return tensor;
}

Result<cv::Mat> CopyToCvMat(const Tensor& tensor) {
  if (!tensor.defined()) return Status::InvalidArgument("tensor is undefined");
  if (tensor.buffer()->kind() != MemoryKind::kHost) {
    return Status::Unsupported("CopyToCvMat requires host memory; copy to CPU first");
  }
  const int rank = tensor.shape().Rank();
  if (rank != 2 && rank != 3) {
    return Status::InvalidArgument("CopyToCvMat requires an HW or HWC tensor");
  }
  const int depth = DTypeToDepth(tensor.dtype());
  if (depth < 0) {
    return Status::Unsupported("cv::Mat conversion supports u8, f32 and f64 only");
  }
  const int64_t rows = tensor.shape().Dim(0);
  const int64_t cols = tensor.shape().Dim(1);
  const int64_t channels = rank == 2 ? 1 : tensor.shape().Dim(2);
  if (rows > std::numeric_limits<int>::max() ||
      cols > std::numeric_limits<int>::max() || channels > CV_CN_MAX) {
    return Status::Unsupported("tensor dimensions exceed cv::Mat limits");
  }

  cv::Mat result(static_cast<int>(rows), static_cast<int>(cols),
                 CV_MAKETYPE(depth, static_cast<int>(channels)));
  if (tensor.IsContiguous()) {
    std::memcpy(result.data, tensor.data(), tensor.ByteSize());
  } else {
    auto contiguous = tensor.Clone();
    if (!contiguous.ok()) return contiguous.status();
    std::memcpy(result.data, contiguous.value().data(), tensor.ByteSize());
  }
  return result;
}

}  // namespace mycv
