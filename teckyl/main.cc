#include <fstream>
#include <iostream>
#include <map>
#include <set>

#include "teckyl/tc/lang/parser.h"
#include "teckyl/tc/lang/sema.h"
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Module.h>
#include <mlir/Dialect/StandardOps/EDSC/Intrinsics.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include "mlir/Dialect/SCF/SCF.h"

#include "teckyl/MLIRGen.h"

// Commandline options
static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                  llvm::cl::init("-"), llvm::cl::value_desc("filename"));
enum Action { None, DumpAST, DumpMLIR, DumpInference };

static llvm::cl::opt<enum Action> emitAction(
    "emit", llvm::cl::desc("Select the kind of output desired"),
    llvm::cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    llvm::cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    llvm::cl::values(clEnumValN(DumpInference, "inference",
                                "output inference results")));

static llvm::cl::opt<teckyl::MLIRGenOptions::BodyOp> bodyOp(
    "body-op",
    llvm::cl::desc("Select the operation used for the body of computations"),
    llvm::cl::values(clEnumValN(teckyl::MLIRGenOptions::BodyOp::LinalgGeneric,
                                "linalg.generic", "Linalg.generic")),
    llvm::cl::values(clEnumValN(teckyl::MLIRGenOptions::BodyOp::ScfFor,
                                "scf.for",
                                "Sets of nested instances of Scf.for")));

static llvm::cl::opt<bool> specializeLinalgOps(
    "specialize-linalg-ops",
    llvm::cl::desc("Use structured Ops from the linalg dialect for common "
                   "operation (e.g., matrix multiplications)"),
    llvm::cl::init(false));

// Reads an entire file into a string
std::string readFile(const std::string &filename) {
  std::ifstream ifs(filename);

  return std::string((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());
}

// Parses a string with TCs and returns a map with one entry for each
// kernel, composed of the kernel's name and its AST.
std::map<std::string, lang::Def> parse(const std::string &tc,
                                       const std::string &filename) {
  lang::Parser parser(tc, filename);
  std::map<std::string, lang::Def> parsed;

  while (parser.L.cur().kind != lang::TK_EOF) {
    auto t = parser.parseFunction();
    auto def = lang::Def(t);
    auto name = def.name().name();
    parsed.emplace(std::make_pair(name, def));
  }

  return parsed;
}

// Dumps the AST for a set of kernels to stderr
void dumpAST(const std::map<std::string, lang::Def> &tcs) {
  for (const auto &res : tcs)
    std::cerr << res.second << std::endl;
}

// Dumps the inference results from the semantic analysis for a set of
// kernels to stderr
void dumpInference(const std::map<std::string, lang::Def> &tcs) {
  tc::CompilerOptions co;
  co.printRanges = true;

  lang::Sema sema(co);

  for (const auto &res : tcs)
    auto func = sema.checkFunction(res.second);
}

// Generates an MLIR representation for each TC kernel and dumps a
// textual reprsentation to stderr.
//
// Returns 0 on success or 1 in case of an error.
void dumpMLIR(const std::map<std::string, lang::Def> &tcs) {
  mlir::registerDialect<mlir::StandardOpsDialect>();
  mlir::registerDialect<mlir::linalg::LinalgDialect>();
  mlir::registerDialect<mlir::scf::SCFDialect>();
  mlir::MLIRContext context;
  mlir::ModuleOp module;
  mlir::OpBuilder builder(&context);
  teckyl::MLIRGenOptions options;
  lang::Sema sema;

  options.body_op = bodyOp;
  options.specialize_linalg_ops = specializeLinalgOps;

  if (options.specialize_linalg_ops &&
      options.body_op != teckyl::MLIRGenOptions::BodyOp::LinalgGeneric) {
    THROW_OR_ASSERT(
        teckyl::Exception("--specialize-linalg-ops can only be used in "
                          "conjunction with --body-op=linalg.generic"));
  }

  module = mlir::ModuleOp::create(builder.getUnknownLoc());

  for (auto &tc : tcs) {
    lang::TreeRef checked = sema.checkFunction(tc.second);

    mlir::FuncOp f = teckyl::buildMLIRFunction(context, tc.first,
                                               lang::Def(checked), options);

    module.push_back(f);
  }

  module.dump();

  if (mlir::failed(mlir::verify(module)))
    llvm_unreachable("Module verification error");
}

int main(int argc, char **argv) {
  std::map<std::string, lang::Def> tcs;

  llvm::cl::ParseCommandLineOptions(argc, argv, "teckyl frontend\n");

  std::string source = readFile(inputFilename);

  tcs = parse(source, inputFilename);

#ifdef COMPILE_WITH_EXCEPTIONS
  try {
#endif // COMPILE_WITH_EXCEPTIONS

    switch (emitAction) {
    case Action::DumpAST:
      dumpAST(tcs);
      break;
    case Action::DumpMLIR:
      dumpMLIR(tcs);
      break;
    case Action::DumpInference:
      dumpInference(tcs);
      break;
    default:
      THROW_OR_ASSERT(teckyl::Exception("Unknown action"));
    }

#ifdef COMPILE_WITH_EXCEPTIONS
  } catch (teckyl::Exception &e) {
    std::cerr << "Error: " << e.getMessage() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "An unknown error has occured." << std::endl;
    return 1;
  }
#endif // COMPILE_WITH_EXCEPTIONS
}
