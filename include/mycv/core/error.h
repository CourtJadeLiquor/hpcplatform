#ifndef MYCV_CORE_ERROR_H_
#define MYCV_CORE_ERROR_H_

#include <string>
#include <utility>

namespace mycv {

enum class StatusCode : int {
  kOk = 0,
  kInvalidArgument = 1,
  kOutOfMemory = 2,
  kUnsupported = 3,
  kNotImplemented = 4,
  kBadState = 5,
  kInternal = 6,
};

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

  bool ok() const { return code_ == StatusCode::kOk; }
  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }

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
  // NOLINTNEXTLINE(google-explicit-constructor): mirrors std::expected ergonomics.
  Result(T value) : value_(std::move(value)), ok_(true) {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  Result(Status status) : status_(std::move(status)), ok_(false) {}

  bool ok() const { return ok_; }
  const Status& status() const { return status_; }
  T& value() { return value_; }
  const T& value() const { return value_; }
  T ValueOr(T fallback) const { return ok_ ? value_ : std::move(fallback); }

 private:
  T value_{};
  Status status_;
  bool ok_ = false;
};

}  // namespace mycv

// Propagates a failed Status out of a function returning Status or Result<T>.
#define MYCV_RETURN_IF_ERROR(expr)              \
  do {                                          \
    ::mycv::Status mycv_status = (expr);         \
    if (!mycv_status.ok()) return mycv_status;   \
  } while (0)

#endif  // MYCV_CORE_ERROR_H_
