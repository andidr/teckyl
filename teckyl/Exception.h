#ifndef TECKYL_EXCEPTION_H
#define TECKYL_EXCEPTION_H

#include <string>

namespace teckyl {
class Exception {
public:
  Exception(const std::string &msg) : msg(msg) {}
  virtual ~Exception() = default;

  const std::string &getMessage() const { return msg; }

protected:
  const std::string msg;
};
} // namespace teckyl

#endif
