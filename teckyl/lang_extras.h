#ifndef TECKYL_LANG_EXTRAS_H
#define TECKYL_LANG_EXTRAS_H

#include "Exception.h"
#include <tc/lang/tree.h>

#include <map>
#include <set>

namespace teckyl {
// Resursively maps the function `fn` to `tree` and all of its
// descendants in preorder.
static void mapRecursive(const lang::TreeRef &tree,
                         std::function<void(const lang::TreeRef &)> fn) {
  fn(tree);

  for (auto e : tree->trees())
    mapRecursive(e, fn);
}

static bool inline isSignedIntType(int kind) {
  switch (kind) {
  case lang::TK_INT8:
  case lang::TK_INT16:
  case lang::TK_INT32:
  case lang::TK_INT64:
    return true;

  default:
    return false;
  }
}

static inline bool isUnsignedIntType(int kind) {
  switch (kind) {
  case lang::TK_UINT8:
  case lang::TK_UINT16:
  case lang::TK_UINT32:
  case lang::TK_UINT64:
    return true;

  default:
    return false;
  }
}

static inline bool isIntType(int kind) {
  return isSignedIntType(kind) || isUnsignedIntType(kind);
}

static unsigned getIntBits(int kind) {
  switch (kind) {
  case lang::TK_INT8:
  case lang::TK_UINT8:
    return 8;
  case lang::TK_INT16:
  case lang::TK_UINT16:
    return 16;
  case lang::TK_INT32:
  case lang::TK_UINT32:
    return 32;
  case lang::TK_INT64:
  case lang::TK_UINT64:
    return 64;

  default:
    throw mlirgen::Exception("Unexpected kind");
  }
}

static inline bool isFloatType(int kind) {
  switch (kind) {
  case lang::TK_FLOAT:
  case lang::TK_FLOAT16:
  case lang::TK_FLOAT32:
  case lang::TK_FLOAT64:
    return true;

  default:
    return false;
  }
}

using IteratorRangeMap = std::map<std::string, lang::RangeConstraint>;

// Collects all range constraints specified in `where` clauses of the
// comprehension c
static IteratorRangeMap
collectExplicitIteratorBounds(const lang::Comprehension &c) {
  IteratorRangeMap bounds;

  for (auto where : c.whereClauses()) {
    if (where->kind() != lang::TK_RANGE_CONSTRAINT)
      continue;

    auto range = lang::RangeConstraint(where);
    std::string name = range.ident().name();

    bounds.insert({name, range});
  }

  return std::move(bounds);
}

// Collects the set of parameters from the signature of `def` that
// define the sizes of dimensions. for example, for the signature
//
//   def foo(float(M, N) A, float(K) x)
//
// The function would return a set composed of M, N and K.
static std::set<std::string> collectDimSizeParams(const lang::Def &def) {
  std::set<std::string> sizeParams;

  for (const lang::Param &param : def.params()) {
    for (const lang::TreeRef &dim : param.tensorType().dims()) {
      if (dim->kind() == lang::TK_IDENT) {
        lang::Ident ident(dim);

        sizeParams.insert(ident.name());
      }
    }
  }

  return sizeParams;
}

} // namespace teckyl

#endif
