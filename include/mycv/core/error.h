#ifndef MYCV_CORE_ERROR_H_
#define MYCV_CORE_ERROR_H_

#include <string>
#include <utility>

namespace mycv {

// 错误分类：调用方据此决定是修正参数、换后端还是补实现。
enum class StatusCode : int {
  kOk = 0,               // 成功
  kInvalidArgument = 1,  // 参数不合法
  kOutOfMemory = 2,      // 内存分配失败
  kUnsupported = 3,      // 当前构建未启用该能力
  kNotImplemented = 4,   // 能力可用但该组合没有内核
  kBadState = 5,         // 对象状态不允许当前操作
  kInternal = 6,         // 内部不变量被破坏
};

// 返回错误码的稳定名称。
const char* StatusCodeName(StatusCode code);

// Lightweight error object. The library core does not throw exceptions so that
// it stays usable in environments compiled with -fno-exceptions (common on some
// NPU / embedded toolchains).
class Status {
 public:
  Status() = default;
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  static Status Ok() { return Status(); }

  static Status InvalidArgument(std::string message) {
    return Status(StatusCode::kInvalidArgument, std::move(message));
  }
  static Status OutOfMemory(std::string message) {
    return Status(StatusCode::kOutOfMemory, std::move(message));
  }
  static Status Unsupported(std::string message) {
    return Status(StatusCode::kUnsupported, std::move(message));
  }
  static Status NotImplemented(std::string message) {
    return Status(StatusCode::kNotImplemented, std::move(message));
  }
  static Status BadState(std::string message) {
    return Status(StatusCode::kBadState, std::move(message));
  }
  static Status Internal(std::string message) {
    return Status(StatusCode::kInternal, std::move(message));
  }

  // 是否成功。
  bool ok() const { return code_ == StatusCode::kOk; }
  // 错误码；成功时为 kOk。
  StatusCode code() const { return code_; }
  // 人类可读的错误信息。
  const std::string& message() const { return message_; }

  // 格式化为 "code: message"。
  std::string ToString() const;

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

// Minimal Result<T>: either a value or a Status. Intentionally kept small;
// callers must check ok() before touching value().
template <typename T>
class Result {
 public:
  // NOLINTNEXTLINE(google-explicit-constructor): mirrors std::expected
  // ergonomics.
  Result(T value) : value_(std::move(value)), ok_(true) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  Result(Status status) : status_(std::move(status)), ok_(false) {}

  // 是否持有值；为 false 时只能读取 status()。
  bool ok() const { return ok_; }
  // 失败时返回对应 Status；成功时为 kOk。
  const Status& status() const { return status_; }
  // 返回值的引用；调用方必须先确认 ok()。
  T& value() { return value_; }
  const T& value() const { return value_; }
  // 成功时返回值，失败时返回 fallback。
  T ValueOr(T fallback) const { return ok_ ? value_ : std::move(fallback); }

 private:
  T value_{};
  Status status_;
  bool ok_ = false;
};

}  // namespace mycv

// Propagates a failed Status out of a function returning Status or Result<T>.
#define MYCV_RETURN_IF_ERROR(expr)             \
  do {                                         \
    ::mycv::Status mycv_status = (expr);       \
    if (!mycv_status.ok()) return mycv_status; \
  } while (0)

#endif  // MYCV_CORE_ERROR_H_
