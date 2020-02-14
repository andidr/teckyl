#ifndef TECKYL_LANG_EXTRAS_H
#define TECKYL_LANG_EXTRAS_H

#include "Exception.h"
#include <tc/lang/tree.h>

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
} // namespace teckyl

#endif
