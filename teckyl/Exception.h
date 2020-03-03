#ifndef TECKYL_EXCEPTION_H
#define TECKYL_EXCEPTION_H

#include <string>

#ifndef THROW_OR_ASSERT
#ifdef COMPILE_WITH_EXCEPTIONS
#define THROW_OR_ASSERT(X) throw X
#else
#define THROW_OR_ASSERT(X) llvm_unreachable(X.what())
#endif // COMPILE_WITH_EXCEPTIONS
#endif // THROW_OR_ASSERT

namespace teckyl {
class Exception {
public:
  Exception(const std::string &msg) : msg(msg) {}
  virtual ~Exception() = default;

  const std::string &getMessage() const { return msg; }
  const char *what() const noexcept { return msg.c_str(); }

protected:
  const std::string msg;
};
} // namespace teckyl

#endif
