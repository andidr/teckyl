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

// Resursively maps the function `fn` to `tree` and all of its
// descendants in preorder until `fn` evaluates to `false`.
//
// Returns `true` if all invocations of `fn` returned `true`,
// otherwise `false`.
static bool mapRecursiveWhile(const lang::TreeRef &tree,
                              std::function<bool(const lang::TreeRef &)> fn) {
  if (!fn(tree))
    return false;

  for (auto e : tree->trees())
    if (!mapRecursiveWhile(e, fn))
      return false;

  return true;
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
    llvm_unreachable("Unexpected kind");
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

  return bounds;
}

// Collects the set of parameters from the signature of `def` that
// define the sizes of dimensions. for example, for the signature
//
//   def foo(float(M, N) A, float(K) x) -> (float(P, Q) D)
//
// The function would return a set composed of M, N, K, P and Q.
static std::set<std::string> collectDimSizeParams(const lang::Def &def) {
  std::set<std::string> sizeParams;

  auto collectFromParam = [&](const lang::Param &param) {
    for (const lang::TreeRef &dim : param.tensorType().dims()) {
      if (dim->kind() == lang::TK_IDENT) {
        lang::Ident ident(dim);

        sizeParams.insert(ident.name());
      }
    }
  };

  for (const lang::Param &param : def.params())
    collectFromParam(param);

  for (const lang::Param &param : def.returns())
    collectFromParam(param);

  return sizeParams;
}

// Checks if two identifiers have the same name
static inline bool compareIdentifiers(const lang::Ident &a,
                                      const lang::Ident &b) {
  return a.name() == b.name();
}

// Checks if the value of a numeric constant is zero.
static inline bool isZeroConstant(const lang::Const &c) {
  switch (c.type()->kind()) {
  case lang::TK_INT8:
  case lang::TK_INT16:
  case lang::TK_INT32:
  case lang::TK_INT64:
    return c.value<int64_t>() == 0;
  case lang::TK_UINT8:
  case lang::TK_UINT16:
  case lang::TK_UINT32:
  case lang::TK_UINT64:
  case lang::TK_SIZET:
    return c.value<uint64_t>() == 0;
  case lang::TK_FLOAT16:
  case lang::TK_FLOAT32:
  case lang::TK_FLOAT64:
    return c.value<double>() == 0.0;
  }

  llvm_unreachable("Cannot check if constant is zero: unknown constant type");
}

// Checks if an expression `t` is a numeric constant whose value is
// zero.
static inline bool isZeroExpr(const lang::TreeRef &t) {
  return t->kind() == lang::TK_CONST && isZeroConstant(lang::Const(t));
}

// Checks if two constants are equal in both type and value.
static inline bool compareConstants(const lang::Const &a,
                                    const lang::Const &b) {
  if (a.type()->kind() != b.type()->kind())
    return false;

  int kind = a.type()->kind();

  switch (kind) {
  case lang::TK_INT8:
  case lang::TK_INT16:
  case lang::TK_INT32:
  case lang::TK_INT64:
    return a.value<int64_t>() == b.value<int64_t>();
  case lang::TK_UINT8:
  case lang::TK_UINT16:
  case lang::TK_UINT32:
  case lang::TK_UINT64:
    return a.value<uint64_t>() == b.value<uint64_t>();
  case lang::TK_FLOAT16:
  case lang::TK_FLOAT32:
  case lang::TK_FLOAT64:
    return a.value<double>() == b.value<double>();
  }

  llvm_unreachable("Comparing constants of unknown type");
}

// Checks if two expressions either reference the same numeric
// constant or the same symbolic parameter.
static inline bool compareConstOrParamExpr(const lang::TreeRef &a,
                                           const lang::TreeRef &b) {
  return (a->kind() == lang::TK_IDENT && b->kind() == lang::TK_IDENT &&
          compareIdentifiers(lang::Ident(a), lang::Ident(b))) ||
         (a->kind() == lang::TK_CONST && b->kind() == lang::TK_CONST &&
          compareConstants(lang::Const(a), lang::Const(b)));
}

// Checks that each of the specified iterators is used at least once
// for direct indexing (i.e., the iterator is used directly to index a
// tensor dimension) a sub-expression of `e`.
static inline bool
allIteratorsIndexTensorDimension(const std::set<std::string> &iterators,
                                 const lang::TreeRef &e) {
  std::set<std::string> directIterators;

  mapRecursive(e, [&](const lang::TreeRef &t) {
    if (t->kind() == lang::TK_APPLY) {
      const lang::Apply apply(t);

      for (const lang::TreeRef &idx : apply.arguments()) {
        if (idx->kind() == lang::TK_IDENT) {
          directIterators.insert(lang::Ident(idx).name());
        }
      }
    }
  });

  return std::includes(directIterators.begin(), directIterators.end(),
                       iterators.begin(), iterators.end());
}

// Checks if the domain of a single iterator matches the size of a
// tensor dimension it directly indexes
static inline bool iteratorDomainMatchesTensorDimension(
    const std::map<const std::string, lang::TensorType> &paramSpecs,
    const IteratorRangeMap &bounds, const std::string &iterator,
    const std::string &tensor, size_t tensorDim) {
  lang::RangeConstraint range = bounds.at(iterator);
  lang::TreeRef dimSize = paramSpecs.at(tensor).dims()[tensorDim];

  // Must start at zero and end at the size of the dimension (which
  // is either a symbolic constant or a numeric value).
  return isZeroExpr(range.start()) &&
         compareConstOrParamExpr(range.end(), dimSize);
}

// Checks that the domain of each iterator from `indexes` provided in
// `bounds` used for indexing `tensorName` on the LHS of a
// comprehension matches the size of the output tensor dimension it
// indexes specified in `paramSpecs`.
static inline bool comprehensionLHSIteratorDomainsMatchTensorDimensions(
    const std::map<const std::string, lang::TensorType> &paramSpecs,
    const IteratorRangeMap &bounds, const std::string &tensorName,
    const lang::ListView<lang::Ident> &indexes) {
  // Check indexing of the output tensor
  size_t i = 0;
  for (const lang::Ident &idx : indexes) {
    if (!iteratorDomainMatchesTensorDimension(paramSpecs, bounds, idx.name(),
                                              tensorName, i)) {
      return false;
    }

    i++;
  }

  return true;
}

// Checks that the domain specified in by a where clause of `c` of
// each iterator that is used at least once for direct indexing of a
// tensor dimension matches the size of the indexed dimension
// specified in a tensor specifications of `paramSpecs`.
//
// That is, the range must start with 0 and end at the size of the
// tensor dimension.
static inline bool directIteratorDomainsMatchTensorDimensions(
    const lang::Comprehension &c,
    const std::map<const std::string, lang::TensorType> &paramSpecs) {
  IteratorRangeMap bounds = collectExplicitIteratorBounds(c);

  // Check indexing of the output tensor
  if (!comprehensionLHSIteratorDomainsMatchTensorDimensions(
          paramSpecs, bounds, c.ident().name(), c.indices())) {
    return false;
  }

  // Check indexing of the input tensors
  return mapRecursiveWhile(c.rhs(), [&](const lang::TreeRef &e) {
    if (e->kind() == lang::TK_APPLY) {
      lang::Apply apply(e);

      size_t i = 0;
      for (const lang::TreeRef &arg : apply.arguments()) {
        if (arg->kind() == lang::TK_IDENT) {
          if (!iteratorDomainMatchesTensorDimension(paramSpecs, bounds,
                                                    lang::Ident(arg).name(),
                                                    apply.name().name(), i)) {
            return false;
          }
        }

        i++;
      }
    }
    return true;
  });
}

} // namespace teckyl

#endif
