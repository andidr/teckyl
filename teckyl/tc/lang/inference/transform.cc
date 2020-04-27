#include <fstream>
#include <iostream>

#include "teckyl/tc/lang/inference/expr.h"
#include "teckyl/tc/lang/inference/expression_parser.h"
#include "teckyl/tc/lang/inference/transformation.h"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>

// Commandline options
static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"),
                  llvm::cl::init("-"), llvm::cl::value_desc("filename"));

enum Action { None, Distribute, SignConvert, Normalize };

static llvm::cl::opt<enum Action> trafoAction(
    "trafo", llvm::cl::desc("Select the desired transformation"),
    llvm::cl::init(None),
    llvm::cl::values(clEnumValN(None, "none",
                                "no transformation "
                                "(i.e. the identity transformation")),
    llvm::cl::values(clEnumValN(Distribute, "distr",
                                "distribute multiplication over sums")),
    llvm::cl::values(clEnumValN(SignConvert, "sign-conv",
                                "move signs as far into arguments "
                                "of multiplications as possible")),
    llvm::cl::values(clEnumValN(Normalize, "norm",
                                "normalize expression (i.e. convert "
                                "into sum of monomials)")));

enum Assoc { Left, Right };

static llvm::cl::opt<enum Assoc> trafoAssoc(
    "assoc",
    llvm::cl::desc(
        "Select which way to associate operations in normalized expressions"),
    llvm::cl::init(Left),
    llvm::cl::values(clEnumValN(Left, "left",
                                "associate operations to the left")),
    llvm::cl::values(clEnumValN(Right, "right",
                                "associate operations to the right")));

std::string readIfstream(std::istream &ifs) {
  return std::string((std::istreambuf_iterator<char>(ifs)),
                     std::istreambuf_iterator<char>());
}

// Reads an entire file into a string
std::string readFile(const std::string &filename) {
  std::ifstream ifs(filename);

  return readIfstream(ifs);
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "testing of expression infrastructure\n");

  std::string source =
      (inputFilename == "-") ? readIfstream(std::cin) : readFile(inputFilename);

  teckyl::ranges::ExprParser Parser(source);

  const teckyl::ranges::ExprRef expr = Parser.parse();

  teckyl::ranges::ExprRef result;
  switch (trafoAction) {
  case None: {
    teckyl::ranges::Identity trafo;
    result = trafo.run(expr);
    break;
  }
  case Distribute: {
    teckyl::ranges::Distribution trafo;
    result = trafo.run(expr);
    break;
  }
  case SignConvert: {
    teckyl::ranges::SignConversion trafo;
    result = trafo.run(expr);
    break;
  }
  case Normalize: {
    bool leftAssoc = (trafoAssoc == Left);
    auto trafo = teckyl::ranges::Normalization(leftAssoc);
    result = trafo.run(expr);
    break;
  }
  default:
    llvm_unreachable("Unknown action");
  }

  std::cout << (*result) << "\n";

  return 0;
}
