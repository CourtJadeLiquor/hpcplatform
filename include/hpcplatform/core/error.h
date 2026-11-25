#ifndef hpcplatform_CORE_ERROR_H_
#define hpcplatform_CORE_ERROR_H_

#include <string>
#include <utility>

namespace hpcplatform
{

    // Error classification: tells callers whether to fix arguments, switch backends,
    // or add an implementation.
    enum class StatusCode : int
    {
        kOk = 0,               // Success
        kInvalidArgument = 1,  // Invalid argument
        kOutOfMemory = 2,      // Memory allocation failed
        kUnsupported = 3,      // Capability is not enabled in this build
        kNotImplemented = 4,   // No kernel supports this available combination
        kBadState = 5,         // Object state does not allow this operation
        kInternal = 6,         // Internal invariant violated
    };

    // Returns the stable name of a status code.
    const char* StatusCodeName(StatusCode code);

    // Lightweight error object. The library core does not throw exceptions so that
    // it stays usable in environments compiled with -fno-exceptions (common on some
    // NPU / embedded toolchains).
    class Status
    {
    public:
        Status() = default;
        Status(StatusCode code, std::string message) : code_(code), message_(std::move(message)) {}

        static Status Ok() { return Status(); }

        static Status InvalidArgument(std::string message)
        {
            return Status(StatusCode::kInvalidArgument, std::move(message));
        }
        static Status OutOfMemory(std::string message)
        {
            return Status(StatusCode::kOutOfMemory, std::move(message));
        }
        static Status Unsupported(std::string message)
        {
            return Status(StatusCode::kUnsupported, std::move(message));
        }
        static Status NotImplemented(std::string message)
        {
            return Status(StatusCode::kNotImplemented, std::move(message));
        }
        static Status BadState(std::string message)
        {
            return Status(StatusCode::kBadState, std::move(message));
        }
        static Status Internal(std::string message)
        {
            return Status(StatusCode::kInternal, std::move(message));
        }

        // Whether the operation succeeded.
        bool ok() const { return code_ == StatusCode::kOk; }
        // Status code; kOk on success.
        StatusCode code() const { return code_; }
        // Human-readable error message.
        const std::string& message() const { return message_; }

        // Formats as "code: message".
        std::string ToString() const;

    private:
        StatusCode code_ = StatusCode::kOk;
        std::string message_;
    };

    // Minimal Result<T>: either a value or a Status. Intentionally kept small;
    // callers must check ok() before touching value().
    template <typename T>
    class Result
    {
    public:
        // NOLINTNEXTLINE(google-explicit-constructor): mirrors std::expected
        // ergonomics.
        Result(T value) : value_(std::move(value)), ok_(true) {}
        // NOLINTNEXTLINE(google-explicit-constructor)
        Result(Status status) : status_(std::move(status)), ok_(false) {}

        // Whether a value is held; when false, only status() may be read.
        bool ok() const { return ok_; }
        // Returns the failure Status, or an OK Status on success.
        const Status& status() const { return status_; }
        // Returns a reference to the value; caller must check ok() first.
        T& value() { return value_; }
        const T& value() const { return value_; }
        // Returns the value on success, or fallback on failure.
        T ValueOr(T fallback) const { return ok_ ? value_ : std::move(fallback); }

    private:
        T value_{};
        Status status_;
        bool ok_ = false;
    };

}  // namespace hpcplatform

// Propagates a failed Status out of a function returning Status or Result<T>.
// Returns only on failure; the library never prints to stdio by itself.
#define hpcplatform_RETURN_IF_ERROR(expr)                        \
    do                                                           \
    {                                                            \
        const ::hpcplatform::Status hpcplatform_status = (expr); \
        if (!hpcplatform_status.ok()) return hpcplatform_status; \
    } while (0)

#endif  // hpcplatform_CORE_ERROR_H_
