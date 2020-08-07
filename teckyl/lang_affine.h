#ifndef TECKYL_LANG_AFFINE_H
#define TECKYL_LANG_AFFINE_H

#include "teckyl/Exception.h"
#include "teckyl/lang_extras.h"

#include <set>
#include <string>

namespace teckyl {
bool isAffine(const lang::TreeRef &e, const std::set<std::string> &syms);

// Checks whether the identifier refers to a symbol from `syms` or can
// be treated as a constant.
bool isSymbolic(const lang::Ident &ident, const std::set<std::string> &syms) {
  return syms.find(ident.name()) != syms.end();
}

// Checks whether the expression passed in `t` uses at least one
// symbol from `syms`.
bool isSymbolic(const lang::TreeRef &t, const std::set<std::string> &syms) {
  switch (t->kind()) {
  case lang::TK_IDENT:
    return isSymbolic(lang::Ident(t), syms);
  case lang::TK_CONST:
    return false;
  case '+':
  case '-':
  case '*':
  case '/': {
    for (const lang::TreeRef &child : t->trees())
      if (isSymbolic(child, syms))
        return true;

    return false;
  }
  default:;
  }

  Exception err("Unsupported kind '" + lang::kindToString(t->kind()) + "'");
  llvm_unreachable(err.what());
}

// Conservative check whether e is an affine expression with respect
// to the symbols passed in `syms`. The check is conservative in the
// sense that it does not recognize all affine expressions (e.g.,
// `5/(3/i)` is perfectly affine, but is not detected by the check)
// and returns false for cases that cannot be detected reliably.
//
// TODO: Add canonicalization pass
bool isAffine(const lang::TreeRef &e, const std::set<std::string> &syms) {
  switch (e->kind()) {
  case lang::TK_CONST:
    // Only allow integer constants for now
    return isIntType(lang::Const(e).type()->kind());
  case lang::TK_IDENT:
    return true;
  case lang::TK_ACCESS:
    return false;
  case '+':
  case '-': {
    for (const lang::TreeRef &index : e->trees())
      if (!isAffine(index, syms))
        return false;

    return true;
  }
  case '*': {
    unsigned int numSymbolic = 0;

    // At most one factor might be symbolic and must be affine
    for (const lang::TreeRef &child : e->trees()) {
      if (isSymbolic(child, syms)) {
        if (++numSymbolic > 1)
          return false;

        if (!isAffine(child, syms))
          return false;
      }
    }

    return true;
  }

  case '/':
    if (e->trees().size() != 2)
      llvm_unreachable("Division with more than two operands found");

    // Conservatively make sure that the dividend is affine and that
    // there are no symbols in the divisor.
    //
    // TODO: implement canonicalization
    return isAffine(e->tree(0), syms) && !isSymbolic(e->tree(1), syms);

  default:;
  }

  Exception err("Unsupported kind '" + lang::kindToString(e->kind()) + "'");
  llvm_unreachable(err.what());
}

// Conservatively checks whether an expression indexes tensors with
// non-affine expressions wrt. the symbols in `syms`.
//
// Some affine indexing schemes might be recognized as non-affine,
// e.g., `A(1/(1/i))`.
//
// TODO: Canonicalize before checking
bool hasNonAffineIndexing(const lang::TreeRef &e,
                          const std::set<std::string> &syms) {
  switch (e->kind()) {
  case lang::TK_CONST:
    return false;
  case lang::TK_ACCESS: {
    for (const lang::TreeRef &arg : lang::Access(e).arguments())
      if (!isAffine(arg, syms))
        return true;

    return false;
  }
  case '+':
  case '-':
  case '*':
  case '/':
  case '?':
  case '>':
  case '<':
  case lang::TK_GE:
  case lang::TK_LE:
  case lang::TK_EQ: {
    for (const lang::TreeRef &child : e->trees())
      if (hasNonAffineIndexing(child, syms))
        return true;

    return false;
  }
  default:;
  }

  Exception err("Unsupported kind '" + lang::kindToString(e->kind()) + "'");
  llvm_unreachable(err.what());
}
} // namespace teckyl

#endif
