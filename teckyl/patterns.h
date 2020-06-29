#ifndef TECKYL_PATTERNS_H
#define TECKYL_PATTERNS_H

#include "teckyl/tc/lang/tree_views.h"

namespace teckyl {
namespace pattern {

// Checks if a comprehension is a matrix-vector product, i.e., if it
// has the pattern
//
//   C(i) +=! A(i, k) * B(k) or
//   C(i) +=! B(k) * A(i, k)
//
// Returns true if the pattern matches, otherwise false. If
// inverse_operands is non-NULL, its value will be set to false if the
// first pattern matches or true if the second pattern matches. If no
// pattern matches, inverse_operands is left untouched.
static inline bool isMatvecComprehension(const lang::Comprehension &c,
                                         size_t (*canonical_order)[2] = NULL) {
  lang::ListView<lang::Ident> lhsIdents = c.indices();

  // Ensure this is a sum of products
  if (c.assignment()->kind() != lang::TK_PLUS_EQ_B || c.rhs()->kind() != '*')
    return false;

  // Ensure that the output is a vector
  if (lhsIdents.size() != 1)
    return false;

  // Ensure that there are exactly two operands to the multiplication
  if (c.rhs()->trees().size() != 2)
    return false;

  // Ensure that the source operands are indexed tensors
  if (c.rhs()->tree(0)->kind() != lang::TK_ACCESS ||
      c.rhs()->tree(1)->kind() != lang::TK_ACCESS) {
    return false;
  }

  std::vector<lang::Access> accesses = {lang::Access(c.rhs()->tree(0)),
                                        lang::Access(c.rhs()->tree(1))};

  bool inv;

  if (accesses[0].arguments().size() == 2 &&
      accesses[1].arguments().size() == 1) {
    // Matrix is the first operand, vector is the second operand
    inv = false;
  } else if (accesses[0].arguments().size() == 1 &&
             accesses[1].arguments().size() == 2) {
    // Vector is the first operand, matrix is the second operand
    inv = true;
    accesses = {accesses[1], accesses[0]};
  } else {
    // Operands are not exactly one matrix and one vector
    return false;
  }

  // Ensure that the output operand is not used as an input
  if (accesses[0].name().name() == c.ident().name() ||
      accesses[1].name().name() == c.ident().name()) {
    return false;
  }

  // Check that the matrix and vector are indexed directly by identifiers
  if (accesses[0].arguments()[0]->kind() != lang::TK_IDENT ||
      accesses[0].arguments()[1]->kind() != lang::TK_IDENT ||
      accesses[1].arguments()[0]->kind() != lang::TK_IDENT) {
    return false;
  }

  lang::Ident matrixIdents[2] = {lang::Ident(accesses[0].arguments()[0]),
                                 lang::Ident(accesses[0].arguments()[1])};
  lang::Ident vectorIdent = lang::Ident(accesses[1].arguments()[0]);

  // Ensure that the iterator for the output is used to index the
  // first dimension of the input matrix and that the iterator for the
  // second dimension of the input matrix is used to iterate over the
  // input vector
  bool ret = lhsIdents[0].name() == matrixIdents[0].name() &&
             matrixIdents[1].name() == vectorIdent.name();

  if (ret && canonical_order) {
    if (!inv) {
      (*canonical_order)[0] = 0;
      (*canonical_order)[1] = 1;
    } else {
      (*canonical_order)[0] = 1;
      (*canonical_order)[1] = 0;
    }
  }

  return ret;
}

// Checks if a comprehension is a matrix multiplication, i.e., if it
// has the pattern
//
//   C(i, j) +=! A(i, k) * B(k, j) or
//   C(i, j) +=! B(k, j) * A(i, k)
//
// Returns true if the pattern matches, otherwise false. If
// canonical_order is non-NULL, the indexes for the canonical order of
// the input operands will be provided. If no pattern matches,
// canonical_order is left untouched.
static inline bool isMatmulComprehension(const lang::Comprehension &c,
                                         size_t (*canonical_order)[2] = NULL) {
  lang::ListView<lang::Ident> lhsIdents = c.indices();

  // Ensure this is a sum of products
  if (c.assignment()->kind() != lang::TK_PLUS_EQ_B || c.rhs()->kind() != '*')
    return false;

  // Ensure that the output is a matrix
  if (lhsIdents.size() != 2)
    return false;

  // Ensure that there are exactly two operands to the multiplication
  if (c.rhs()->trees().size() != 2)
    return false;

  // Ensure that the source operands are indexed tensors
  if (c.rhs()->tree(0)->kind() != lang::TK_ACCESS ||
      c.rhs()->tree(1)->kind() != lang::TK_ACCESS) {
    return false;
  }

  lang::Access accesses[2] = {lang::Access(c.rhs()->tree(0)),
                              lang::Access(c.rhs()->tree(1))};

  // Ensure that the output operand is not used as an input
  if (accesses[0].name().name() == c.ident().name() ||
      accesses[1].name().name() == c.ident().name()) {
    return false;
  }

  // Ensure that both operands of the multiplication are matrices
  // directly indexed by identifiers
  if (accesses[0].arguments().size() != 2 ||
      accesses[1].arguments().size() != 2 ||
      accesses[0].arguments()[0]->kind() != lang::TK_IDENT ||
      accesses[0].arguments()[1]->kind() != lang::TK_IDENT ||
      accesses[1].arguments()[0]->kind() != lang::TK_IDENT ||
      accesses[1].arguments()[1]->kind() != lang::TK_IDENT) {
    return false;
  }

  // Extract identifiers
  lang::Ident rhsIdents[2][2] = {{lang::Ident(accesses[0].arguments()[0]),
                                  lang::Ident(accesses[0].arguments()[1])},
                                 {lang::Ident(accesses[1].arguments()[0]),
                                  lang::Ident(accesses[1].arguments()[1])}};

  // Check for pattern C(i, j) +=! A(i, k) * B(k, j)
  if (lhsIdents[0].name() == rhsIdents[0][0].name() &&
      rhsIdents[0][1].name() == rhsIdents[1][0].name() &&
      lhsIdents[1].name() == rhsIdents[1][1].name()) {
    if (canonical_order) {
      (*canonical_order)[0] = 0;
      (*canonical_order)[1] = 1;
    }

    return true;
  }
  // Check for pattern C(i, j) +=! B(k, j) * A(i, k)
  else if (lhsIdents[0].name() == rhsIdents[1][0].name() &&
           rhsIdents[1][1].name() == rhsIdents[0][0].name() &&
           lhsIdents[1].name() == rhsIdents[0][1].name()) {
    if (canonical_order) {
      (*canonical_order)[0] = 1;
      (*canonical_order)[1] = 0;
    }

    return true;
  }

  return false;
}
} // namespace pattern
} // namespace teckyl

#endif
