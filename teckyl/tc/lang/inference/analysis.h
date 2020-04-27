#ifndef TECKYL_TC_INFERENCE_ANALYSIS_H
#define TECKYL_TC_INFERENCE_ANALYSIS_H

#include "teckyl/tc/lang/inference/expr.h"

#include <algorithm>
#include <llvm/Support/ErrorHandling.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace teckyl {
namespace ranges {

struct Analysis {
  virtual void reset() {}

  virtual void run(const ExprRef e) = 0;
};

using Value = Constant::value_type;
using ValueVector = std::vector<Value>;
using StringVector = std::vector<std::string>;

struct AtomCollection : public Analysis, ExprVisitor {
  // We consider the following as atoms inside an expression:
  //   (a) constant,
  //   (b) parameters, and
  //   (c) variables.
  // In addition, this analysis also counts the negations
  // inside an expression.

  AtomCollection() { reset(); }

  void reset() final {
    constants.clear();
    parameters.clear();
    variables.clear();
    negations = 0;
  }

  void run(const ExprRef e) final { e->visit(*this); }

  ValueVector getConstants() const { return constants; }
  StringVector getParameters() const { return parameters; }
  StringVector getVariables() const { return variables; }
  unsigned getNegations() const { return negations; }

private:
  ValueVector constants;
  StringVector parameters;
  StringVector variables;
  unsigned negations;

  void visitBinOp(const BinOp *b) final {
    const auto left = b->l;
    const auto right = b->r;

    left->visit(*this);
    right->visit(*this);
  }

  void visitNeg(const Neg *n) final {
    ++negations;
    n->expr->visit(*this);
  }

  void visitConstant(const Constant *c) final { constants.push_back(c->val); }
  void visitParameter(const Parameter *p) final { parameters.push_back(p->n); }
  void visitVariable(const Variable *v) final { variables.push_back(v->n); }
};

struct Coefficient {
  Value positiveFactor; // read: positive part of this coefficient's constant
                        // factor
  Value negativeFactor; // read: negative part of this coefficient's constant
                        // factor hence: the value of this coefficient's
                        // constant factor
                        //        is equal to ('positiveFactor' -
                        //        'negativeFactor')
  StringVector parameters;

  // Normalize a coefficient by sorting its parameters:
  void normalize() { std::sort(parameters.begin(), parameters.end()); }

  // Build an expression that represents this coefficient.
  // In the resulting expression, multiplications associate to the left:
  ExprRef toExprL() const {
    ExprRef expr = preFactorExpr();

    for (auto p : parameters) {
      auto paramExpr = std::make_shared<Parameter>(p);
      expr = std::make_shared<BinOp>(TIMES, expr, paramExpr);
    }

    return expr;
  }

  // Build an expression that represents this coefficient.
  // In the resulting expression, multiplications associate to the right:
  ExprRef toExprR() const {
    ExprRef expr = preFactorExpr();

    StringVector params = parameters;
    std::reverse(params.begin(), params.end());

    for (auto p : params) {
      auto paramExpr = std::make_shared<Parameter>(p);
      expr = std::make_shared<BinOp>(TIMES, paramExpr, expr);
    }

    return expr;
  }

private:
  ExprRef preFactorExpr() const {
    ExprRef expr;
    if (positiveFactor == 0) {
      expr = std::make_shared<Neg>(std::make_shared<Constant>(negativeFactor));
    } else if (negativeFactor == 0) {
      expr = std::make_shared<Constant>(positiveFactor);
    } else {
      expr = std::make_shared<BinOp>(
          MINUS, std::make_shared<Constant>(positiveFactor),
          std::make_shared<Constant>(negativeFactor));
    }
    return expr;
  }
};

struct Monomial {
  std::vector<Coefficient> coefficients;
  StringVector variables;

  // Normalize a monomial by
  //   (a) sorting its variables,
  //   (b) combining coefficients with the same parameters
  //   (c) normalizing all coefficients, and
  //   (d) sorting the (normalized) coefficients by their parameters.
  void normalize() {
    std::sort(variables.begin(), variables.end());

    struct FactorPair {
      Value positive;
      Value negative;

      FactorPair &operator+=(const FactorPair &rhs) {
        positive += rhs.positive;
        negative += rhs.negative;
        return *this;
      }
    };

    std::map<StringVector, FactorPair> combined;

    for (auto c : coefficients) {
      c.normalize();
      // Note that 'c.parameters' are now sorted:
      const auto &params = c.parameters;
      const auto positiveFactor = c.positiveFactor;
      const auto negativeFactor = c.negativeFactor;

      if (combined.count(params) == 0) {
        combined[params] = {positiveFactor, negativeFactor};
      } else {
        combined[params] += {positiveFactor, negativeFactor};
      }
    }

    coefficients.clear();
    // Note that in the following, 'combined' is traversed in the order
    // of its keys (i.e. the parameters). Hence the resulting 'coefficients'
    // vector will be ordered by the parameters of the coefficients.
    for (auto cc : combined) {
      coefficients.push_back(
          {cc.second.positive, cc.second.negative, cc.first});
      // Note that 'cc.first' is a sorted vector of parameters.
      // Hence the resulting coefficients are already normalized.
    }
  }

  // Build an expression that represents this monomial.
  // In the resulting expression, multiplications associate to the left:
  ExprRef toExprL() const {
    if (coefficients.size() == 0)
      llvm_unreachable("Monomial must have at least one coefficient");

    ExprRef expr = coefficients[0].toExprL();
    for (auto c = coefficients.begin() + 1; c != coefficients.end(); c++) {
      expr = std::make_shared<BinOp>(PLUS, expr, c->toExprL());
    }

    for (auto v : variables) {
      auto varExpr = std::make_shared<Variable>(v);
      expr = std::make_shared<BinOp>(TIMES, expr, varExpr);
    }

    return expr;
  }

  // Build an expression that represents this monomial.
  // In the resulting expression, multiplications associate to the left:
  ExprRef toExprR() const {
    if (coefficients.size() == 0)
      llvm_unreachable("Monomial must have at least one coefficient");

    std::vector<Coefficient> coeffs = coefficients;
    std::reverse(coeffs.begin(), coeffs.end());

    ExprRef expr = coeffs[0].toExprR();
    for (auto c = coeffs.begin() + 1; c != coeffs.end(); c++) {
      expr = std::make_shared<BinOp>(PLUS, c->toExprR(), expr);
    }

    StringVector vars = variables;
    std::reverse(vars.begin(), vars.end());

    for (auto v : vars) {
      auto varExpr = std::make_shared<Variable>(v);
      expr = std::make_shared<BinOp>(TIMES, varExpr, expr);
    }

    return expr;
  }
};

struct MonomialCollection : public Analysis, ExprVisitor {
  MonomialCollection() { reset(); }

  void reset() final { monomials.clear(); }

  void run(const ExprRef e) final { e->visit(*this); }

  std::vector<Monomial> get() {
    // Returns normalized monomials.

    std::vector<Monomial> result;

    for (auto m : monomials) {
      Monomial mono = {m.second, m.first};
      mono.normalize();
      result.push_back(mono);
    }

    return result;
  }

private:
  std::map<StringVector, std::vector<Coefficient>> monomials;

  const StringVector empty;

  AtomCollection AC;

  void visitBinOp(const BinOp *b) final {
    const auto left = b->l;
    const auto right = b->r;

    if (!b->isMonomialExpr()) {
      left->visit(*this);
      right->visit(*this);
      return;
    }

    // Expression 'b' is a monomial; so collect its atoms.
    AC.reset();
    AC.run(std::make_shared<BinOp>(*b));

    StringVector vars = AC.getVariables();
    std::sort(vars.begin(), vars.end());

    Value factor = 1;
    for (const auto c : AC.getConstants()) {
      factor *= c;
    }

    const Value positiveFactor = (AC.getNegations() & 1) ? 0 : factor;
    const Value negativeFactor = (AC.getNegations() & 1) ? factor : 0;

    const StringVector params = AC.getParameters();

    monomials[vars].push_back({positiveFactor, negativeFactor, params});
  }

  void visitNeg(const Neg *n) final {
    // The logic of 'visitBinOp' ensures that we only
    // get here if we are not inside another monomial.
    const auto savedMonomials = monomials;

    monomials.clear();
    n->expr->visit(*this);
    auto monomials_in_expr = monomials;

    monomials = savedMonomials;

    for (auto m : monomials_in_expr) {
      // Note that variables in the monomials obtained by the call to
      // 'n->expr->visit(...)' can be assumed to be sorted (cf. the
      // handling of the local variable 'vars' inside method 'visitBinOp').
      const StringVector vars = m.first;

      // Since we are under a 'Neg', swap the positive and negative factors
      // in the coefficients of the monomials in 'monomials_in_expr':
      for (auto c : m.second) {
        const Value tmp = c.negativeFactor;
        c.negativeFactor = c.positiveFactor;
        c.positiveFactor = tmp;

        monomials[vars].push_back(c);
      }
    }
  }

  void visitConstant(const Constant *c) final {
    // The logic of 'visitBinOp' ensures that we only
    // get here if we are not inside another monomial.
    monomials[empty].push_back({c->val, 0, empty});
  }

  void visitParameter(const Parameter *p) final {
    // The logic of 'visitBinOp' ensures that we only
    // get here if we are not inside another monomial.
    StringVector params = {p->n};
    monomials[empty].push_back({1, 0, params});
  }

  void visitVariable(const Variable *v) final {
    // The logic of 'visitBinOp' ensures that we only
    // get here if we are not inside another monomial.
    StringVector vars = {v->n};
    monomials[vars].push_back({1, 0, empty});
  }
};

} // namespace ranges
} // namespace teckyl

#endif // TECKYL_TC_INFERENCE_ANALYSIS_H
