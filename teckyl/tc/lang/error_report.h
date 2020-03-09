/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TECKYL_TC_LANG_ERROR_REPORT_H_
#define TECKYL_TC_LANG_ERROR_REPORT_H_

#include "teckyl/tc/lang/tree.h"
#include "teckyl/tc/utils/compiler_options.h"

#ifndef THROW_OR_ASSERT
#ifdef COMPILE_WITH_EXCEPTIONS
#define THROW_OR_ASSERT(X) throw X
#else
#define THROW_OR_ASSERT(X) llvm_unreachable(X.what())
#endif // COMPILE_WITH_EXCEPTIONS
#endif // THROW_OR_ASSERT

namespace lang {

#ifdef COMPILE_WITH_EXCEPTIONS
struct ErrorReport : public std::exception {
#else
struct ErrorReport {
#endif // COMPILE_WITH_EXCEPTIONS
  ErrorReport(const ErrorReport &e)
      : ss(e.ss.str()), context(e.context), the_message(e.the_message) {}

  ErrorReport(TreeRef context) : context(context->range()) {}
  ErrorReport(SourceRange range) : context(std::move(range)) {}
  const char *what() const noexcept {
    std::stringstream msg;
    msg << "\n" << ss.str() << ":\n";
    context.highlight(msg);
    the_message = msg.str();
    return the_message.c_str();
  }

private:
  template <typename T>
  friend const ErrorReport &operator<<(const ErrorReport &e, const T &t);

  mutable std::stringstream ss;
  SourceRange context;
  mutable std::string the_message;
};

inline void
warn(const ErrorReport &err,
     const tc::CompilerOptions &compilerOptions = tc::CompilerOptions()) {
  if (compilerOptions.emitWarnings) {
    std::cerr << "WARNING: " << err.what();
  }
}

template <typename T>
const ErrorReport &operator<<(const ErrorReport &e, const T &t) {
  e.ss << t;
  return e;
}

#define TC_ASSERT(ctx, cond)                                                   \
  if (!(cond)) {                                                               \
    ::lang::ErrorReport err(ctx);                                              \
    err << __FILE__ << ":" << __LINE__ << ": assertion failed: " << #cond;     \
    THROW_OR_ASSERT(err);                                                      \
  }
} // namespace lang

#endif // TECKYL_TC_LANG_ERROR_REPORT_H_
