#ifndef TECKYL_MLIRAFFINEEXPRGEN_H
#define TECKYL_MLIRAFFINEEXPRGEN_H

#include "Exception.h"
#include "lang_extras.h"

#include <mlir/IR/AffineExpr.h>

#include <map>

namespace teckyl {

// Translates affine tensor expressions to mlir::AffineExpr. The
// generator blindly translates sub-expressions without performing any
// checks. The caller must ensure that the expression to be translated
// is indeed affine, otherwise the generator might trigger an
// assertion.
class MLIRAffineExprGen {
public:
  MLIRAffineExprGen(mlir::MLIRContext *context,
                    const std::map<std::string, unsigned int> &iteratorDims)
      : iteratorDims(iteratorDims), context(context) {}

  // Builds an AffineExpr for each of the arguments of `apply` and
  // returns the result in a vector.
  std::vector<mlir::AffineExpr>
  buildAffineExpressions(const lang::Apply &apply) {
    std::vector<mlir::AffineExpr> res;

    for (const lang::TreeRef &idxExpr : apply.arguments())
      res.push_back(buildAffineExpression(idxExpr));

    return res;
  }

  // Builds an AffineExpr for each of the identifiers and returns the
  // result in a vector.
  std::vector<mlir::AffineExpr>
  buildAffineExpressions(const lang::ListView<lang::Ident> &idents) {
    std::vector<mlir::AffineExpr> res;

    for (const lang::Ident &ident : idents)
      res.push_back(buildAffineExpression(ident));

    return res;
  }

  // Builds an AffineExpr for a tensor expression `t`
  mlir::AffineExpr buildAffineExpression(const lang::TreeRef &t) {
    switch (t->kind()) {
    case lang::TK_IDENT: {
      lang::Ident ident(t);
      unsigned int iterDimIdx = iteratorDims.at(ident.name());
      return mlir::getAffineDimExpr(iterDimIdx, context);
    }
    case lang::TK_CONST: {
      lang::Const cst(t);
      int tKind = cst.type()->kind();

      if (!isIntType(tKind))
        throw Exception("Constant is not an integer");

      // FIXME: AffineExpr uses *signed* 64-bit integers for
      // constants, so the *unsigned* constants from TC cannot
      // necessarily be respresented correctly. Bail out if the TC
      // constant is too big.
      if (tKind == lang::TK_UINT64) {
        uint64_t uintval = cst.value<uint64_t>();

        if (uintval >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
          throw Exception("Unsigned integer constant too big");
      }

      return mlir::getAffineConstantExpr(cst.value<int64_t>(), context);
    }
    case '+':
      return buildAffineBinaryExpression(t, mlir::AffineExprKind::Add);
    case '-':
      return buildAffineSubtraction(t);
    case '*':
      return buildAffineBinaryExpression(t, mlir::AffineExprKind::Mul);
    case '/':
      return buildAffineBinaryExpression(t, mlir::AffineExprKind::FloorDiv);
    default:
      throw Exception("Unsupported operator for affine expression");
    }
  }

protected:
  const std::map<std::string, unsigned int> &iteratorDims;
  mlir::MLIRContext *context;

  mlir::AffineExpr buildAffineBinaryExpression(const lang::TreeRef &t,
                                               mlir::AffineExprKind kind) {
    if (t->trees().size() != 2)
      throw Exception("Binary expression with an operator count != 2");

    mlir::AffineExpr lhs = buildAffineExpression(t->tree(0));
    mlir::AffineExpr rhs = buildAffineExpression(t->tree(1));
    return mlir::getAffineBinaryOpExpr(kind, lhs, rhs);
  }

  // There re no subtraction expressions for AffineExpr; emulate by
  // creating an addition with -1 as a factor for the second operand.
  mlir::AffineExpr buildAffineSubtraction(const lang::TreeRef &t) {
    if (t->trees().size() != 2)
      throw Exception("Subtraction expression with an operator count != 2");

    mlir::AffineExpr lhs = buildAffineExpression(t->tree(0));
    mlir::AffineExpr rhsSub = buildAffineExpression(t->tree(1));
    mlir::AffineExpr minusOne = mlir::getAffineConstantExpr(-1, context);
    mlir::AffineExpr rhs = mlir::getAffineBinaryOpExpr(
        mlir::AffineExprKind::Mul, minusOne, rhsSub);

    return mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::Add, lhs, rhs);
  }
};
} // namespace teckyl

#endif
