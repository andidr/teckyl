#ifndef TECKYL_MLIRGEN_H
#define TECKYL_MLIRGEN_H

#include "teckyl/Exception.h"

#include "teckyl/tc/lang/tree_views.h"
#include <mlir/IR/Function.h>

#include <sstream>

namespace teckyl {
namespace mlirgen {

class Exception : public teckyl::Exception {
public:
  Exception(const std::string &msg) : teckyl::Exception(msg) {}
};

class SourceException : public Exception {
public:
  SourceException(const mlir::FileLineColLoc &l, const std::string &msg)
      : Exception(buildMessage(l, msg)) {}

private:
  static std::string buildMessage(const mlir::FileLineColLoc &l,
                                  const std::string &msg) {
    std::stringstream ss;

    ss << l.getFilename().str() << ":" << l.getLine() << ":" << l.getColumn()
       << ": " << msg;

    return ss.str();
  }
};

} // namespace mlirgen

class MLIRGenOptions {
public:
  enum class BodyOp { LinalgGeneric, ScfFor };

  BodyOp body_op;
};

mlir::FuncOp
buildMLIRFunction(mlir::MLIRContext &context, const std::string &name,
                  const lang::Def &tc,
                  const MLIRGenOptions &options = MLIRGenOptions{});

} // namespace teckyl

#endif
