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
enum Action { None, DumpAST, DumpMLIR };

static llvm::cl::opt<enum Action> emitAction(
    "emit", llvm::cl::desc("Select the kind of output desired"),
    llvm::cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    llvm::cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")));

static llvm::cl::opt<bool> forceStdLoops(
    "force-std-loops",
    llvm::cl::desc(
        "Force use of standard loops when generating code for comprehensions"),
    llvm::cl::init(false));

// Reads an entire file into a string
std::string readFile(const std::string &filename) {
  std::ifstream ifs(filename);

  return std::string((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());
}

// Parses a string with TCs and returns a map with one entry for each
// kernel, composed of the kernel's name and its AST.
std::map<std::string, lang::Def> parse(const std::string &tc) {
  lang::Parser parser(tc);
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

  options.force_std_loops = forceStdLoops;

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

  tcs = parse(source);

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
