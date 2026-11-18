#include <iostream>

#include <opencv2/core.hpp>

#include "mycv/mycv.h"

int main() {
  using namespace mycv;
  auto context_result = Context::Create(BackendType::kOpenCV);
  if (!context_result.ok()) {
    std::cerr << context_result.status().ToString() << '\n';
    return 1;
  }
  auto context = context_result.value();

  cv::Mat mat(2, 2, CV_8UC3);
  for (int i = 0; i < mat.rows * mat.cols * mat.channels(); ++i) {
    mat.data[i] = static_cast<unsigned char>(i * 10);
  }
  auto input = CopyFromCvMat(*context, mat);
  if (!input.ok()) return 1;

  OpContext ops;
  ops.context = context.get();
  ops.stream = &context->stream();
  auto resized = Resize(ops, input.value(), 4, 4);
  if (!resized.ok()) return 1;
  auto normalized = Normalize(ops, resized.value(), {0, 0, 0}, {255, 255, 255});
  if (!normalized.ok()) return 1;
  auto output = CopyToCvMat(normalized.value());
  if (!output.ok()) return 1;

  std::cout << "OpenCV pipeline: cv::Mat -> Tensor -> resize -> normalize -> cv::Mat\n";
  std::cout << "output: " << output.value().rows << 'x' << output.value().cols
            << "x" << output.value().channels() << ", first=";
  const cv::Vec3f first = output.value().at<cv::Vec3f>(0, 0);
  std::cout << first[0] << ',' << first[1] << ',' << first[2] << '\n';
  return 0;
}
