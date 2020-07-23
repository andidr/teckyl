#include "HeaderGen.h"

#include <sstream>
#include <unordered_set>

namespace teckyl {

// Returns a string describing the C data type for a given scalar lang
// data kind. For integers with less than 8 bits, "void" is
// returned. For unsupported scalar types, the function aborts.
static const char *getCType(int kind) {
  switch (kind) {
  case lang::TK_UINT2:
  case lang::TK_UINT4:
    return "void";
  case lang::TK_UINT8:
    return "uint8_t";
  case lang::TK_UINT16:
    return "uint16_t";
  case lang::TK_UINT32:
    return "uint32_t";
  case lang::TK_UINT64:
    return "uint64_t";

  case lang::TK_INT2:
  case lang::TK_INT4:
    return "void";
  case lang::TK_INT8:
    return "int8_t";
  case lang::TK_INT16:
    return "int16_t";
  case lang::TK_INT32:
    return "int32_t";
  case lang::TK_INT64:
    return "int64_t";

  case lang::TK_SIZET:
    return "size_t";

  case lang::TK_FLOAT:
  case lang::TK_FLOAT32:
    return "float";
  case lang::TK_FLOAT64:
    return "double";
  }

  llvm_unreachable("Unsupported scalar type");
}

// Generate a function signature for a tensor function using
// "flattened" memrefs as parameters (i.e., for a 2d memref "A",
// parameters A_allocatedPtr, A_alignedPtr, A_offset, A_size0,
// A_size1, A_stride0, A_stride1 would be added).
//
// The parameters are listed in order of the tensor function
// definition from left to right with input parameters befoe output
// parameters.
void genMemrefSignature(std::stringstream &ss, lang::Def def) {
  ss << "void " << def.name().name() << "(";

  bool isFirstParam = true;

  auto genParam = [&](lang::Param &param, bool isInput) {
    if (isFirstParam)
      isFirstParam = false;
    else
      ss << ", ";

    if (isInput)
      ss << "const ";

    ss << getCType(param.tensorType().scalarType()) << "* "
       << param.ident().name() << "_allocatedPtr, ";

    if (isInput)
      ss << "const ";

    ss << getCType(param.tensorType().scalarType()) << "* "
       << param.ident().name() << "_alignedPtr, "
       << "int64_t " << param.ident().name() << "_offset";

    for (int i = 0; i < param.tensorType().dims().size(); i++)
      ss << ", int64_t " << param.ident().name() << "_size" << i;

    for (int i = 0; i < param.tensorType().dims().size(); i++)
      ss << ", int64_t " << param.ident().name() << "_stride" << i;
  };

  for (lang::Param inParam : def.params())
    genParam(inParam, true);

  for (lang::Param outParam : def.returns())
    genParam(outParam, false);

  ss << ");" << std::endl;
}

// Generates wrapper function for a tensor function using only bare
// pointers and the necessary parameters for parametric
// dimensions. The generated function calls the original function with
// appropriate memref parameters for offsets (always 0), sizes
// (derived from size parameters or constants if defined statically),
// and strides in row-major format.
//
// The name of the generated function is the original name with the
// suffix "_wrap". The parameters are listed in order of the tensor
// function definition from left to right with pointers for input
// parameters first, pointers for output parameters second and symbols
// for parametric dimensions last in order of their appearance.
//
// E.g., for the following input, definition
//
//   def mm(float(M,128) A, float(128,N) B) -> (float(M,N) C) { ... }
//
// this generates a wrapper function with the following signature:
//
//   static inline void
//   mm_wrap(const float* A, const float* B,
//           float* C,
//           uint64_t M, uint64_t N)
//
void genParamWrapper(std::stringstream &ss, lang::Def def) {
  std::unordered_set<std::string> sizeParams;
  std::vector<std::string> sizeParamsSeq;

  ss << "static inline void " << def.name().name() << "_wrap(";

  bool isFirstParam = true;

  auto genParam = [&](lang::Param param, bool isInput) {
    if (isFirstParam)
      isFirstParam = false;
    else
      ss << ", ";

    if (isInput)
      ss << "const ";

    ss << getCType(param.tensorType().scalarType()) << "* "
       << param.ident().name();
    for (const lang::TreeRef &dim : param.tensorType().dims()) {
      if (dim->kind() == lang::TK_IDENT) {
        lang::Ident ident(dim);

        if (sizeParams.insert(ident.name()).second)
          sizeParamsSeq.push_back(ident.name());
      }
    }
  };

  for (lang::Param inParam : def.params())
    genParam(inParam, true);

  for (lang::Param outParam : def.returns())
    genParam(outParam, false);

  for (const std::string &sizeParamName : sizeParamsSeq)
    ss << ", uint64_t " << sizeParamName;

  ss << ") {" << std::endl;

  bool isFirstArg = true;

  auto genMemrefArgs = [&](const lang::Param &param) {
    if (isFirstArg)
      isFirstArg = false;
    else
      ss << ", ";

    ss << param.ident().name() << ", " << param.ident().name() << ", "
       << "0";

    lang::ListView<lang::TreeRef> dims = param.tensorType().dims();

    // Sizes
    for (const lang::TreeRef &dim : dims) {
      ss << ", ";

      if (dim->kind() == lang::TK_IDENT) {
        lang::Ident ident(dim);
        ss << ident.name();
      } else if (dim->kind() == lang::TK_CONST) {
        lang::Const cst(dim);
        ss << cst.value();
      }
    }

    // Strides
    for (size_t i = 0; i < dims.size(); i++) {
      if (i == dims.size() - 1)
        ss << ", 1";
      else
        ss << ", ";

      for (size_t j = i + 1; j < dims.size(); j++) {
        lang::TreeRef dim = dims[j];

        if (j > i + 1)
          ss << "*";

        if (dim->kind() == lang::TK_IDENT) {
          lang::Ident ident(dim);
          ss << ident.name();
        } else if (dim->kind() == lang::TK_CONST) {
          lang::Const cst(dim);
          ss << cst.value();
        }
      }
    }
  };

  ss << "\t" << def.name().name() << "(";

  for (const lang::Param &inParam : def.params())
    genMemrefArgs(inParam);

  for (const lang::Param &outParam : def.returns())
    genMemrefArgs(outParam);

  ss << ");" << std::endl << "}" << std::endl;
}

// Generate a C99 header file with the signatures for the functions
// given in tcs. The parameter includeGuard is the preprocessor symbol
// used to protect the generated header file against double inclusion.
std::string genHeader(const std::map<std::string, lang::Def> &tcs,
                      const std::string &includeGuard) {
  std::stringstream ss;

  ss << "#ifndef " << includeGuard << std::endl
     << "#define " << includeGuard << std::endl
     << std::endl
     << "#include <stdint.h>" << std::endl
     << "#include <stdlib.h>" << std::endl
     << std::endl;

  for (const std::pair<std::string, lang::Def> &def : tcs) {
    genMemrefSignature(ss, def.second);
    ss << std::endl;
    genParamWrapper(ss, def.second);
  }

  ss << std::endl;
  ss << "#endif /* " << includeGuard << " */" << std::endl;

  return ss.str();
}
} // namespace teckyl
