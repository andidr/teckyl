#ifndef TECKYL_TC_INFERENCE_EXPR_H
#define TECKYL_TC_INFERENCE_EXPR_H

#include "teckyl/tc/lang/tree_views.h"

#include <llvm/Support/ErrorHandling.h>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_set>

namespace teckyl {
namespace ranges {

// Kinds of (non-abstract) expressions,
// used for comparison operator '<':
enum ExprKind { EK_BinOp, EK_Neg, EK_Variable, EK_Parameter, EK_Constant };
  
struct Expr;
using ExprRef = std::shared_ptr<Expr>;
  
struct ExprVisitor;
  
// Base class for expressions for range inference
struct Expr {
  explicit Expr(ExprKind k) : kind(k) {}

  virtual bool isConstExpr() const = 0;
  virtual bool isAffineExpr() const = 0;
  virtual bool isSumExpr() const = 0;
  virtual bool isMonomialExpr() const = 0;
  
  virtual bool isBinOp() const = 0;
  virtual bool isNeg() const = 0;
  virtual bool isConstant() const = 0;
  virtual bool isSymbol() const = 0;
  virtual bool isVariable() const = 0;
  virtual bool isParameter() const = 0;
  
  virtual void visit(ExprVisitor &v) const;
  
  virtual std::ostream &print(std::ostream &out) const = 0;

  friend std::ostream &operator<<(std::ostream &out, const Expr &e) {
    return e.print(out);
  }

  virtual bool operator==(const Expr &other) const = 0;
  virtual bool operator<(const Expr &other) const = 0;

  ExprKind getKind() const { return kind; };

  static ExprRef
  fromTreeRef(const lang::TreeRef &t,
              const std::unordered_set<std::string> &rangeParams);

private:
  const ExprKind kind;
};

enum optype { PLUS, MINUS, TIMES };

struct BinOp : public Expr {
  optype op;
  std::shared_ptr<Expr> l, r;

  explicit BinOp(optype operation, std::shared_ptr<Expr> left,
                 std::shared_ptr<Expr> right)
      : Expr(EK_BinOp), op(operation), l(left), r(right) {}

  bool isConstExpr() const final {
    return l->isConstExpr() && r->isConstExpr();
  }

  bool isAffineExpr() const final {
    switch (op) {
    case PLUS:
    case MINUS:
      return l->isAffineExpr() && r->isAffineExpr();
    case TIMES: {
      bool isLeftAffine = l->isAffineExpr() && r->isConstExpr();
      bool isRightAffine = r->isAffineExpr() && l->isConstExpr();
      return isLeftAffine || isRightAffine;
    }
    default:
      llvm_unreachable("Unknown op");
    }
  }

  bool isSumExpr() const final {
    return (op == PLUS) || (op == MINUS);
  }

  bool isMonomialExpr() const final {
    if (op == TIMES)
      return l->isMonomialExpr() && r->isMonomialExpr();
    else
      return false;
  }
  
  bool isBinOp() const final { return true; };
  bool isNeg() const final { return false; };
  bool isConstant() const final { return false; }
  bool isSymbol() const final { return false; }
  bool isVariable() const final { return false; }
  bool isParameter() const final { return false; }

  void visit(ExprVisitor &v) const final;
  
  std::ostream &print(std::ostream &out) const final;

  bool operator==(const Expr &other) const final {
    if (!other.isBinOp())
      return false;

    const auto o = static_cast<const BinOp *>(&other);
    return (this->l == o->l) && (this->op == o->op) && (this->r == o->r);
  }

  bool operator<(const Expr &other) const final {
    if (!other.isBinOp())
      return getKind() < other.getKind();

    const auto o = static_cast<const BinOp *>(&other);

    if (this->l < o->l)
      return true;
    else if (this->l == o->l && this->op < o->op)
      return true;
    else if (this->l == o->l && this->op == o->op && this->r < o->r)
      return true;
    else
      return false;
  }
};

struct Neg : public Expr {
  std::shared_ptr<Expr> expr;

  explicit Neg(std::shared_ptr<Expr> arg) : Expr(EK_Neg), expr(arg) {}

  bool isConstExpr() const final { return expr->isConstExpr(); }

  bool isAffineExpr() const final { return expr->isAffineExpr(); }

  bool isSumExpr() const final { return false; }

  bool isMonomialExpr() const final {
    return expr->isMonomialExpr();
  }
  
  bool isBinOp() const final { return false; };
  bool isNeg() const final { return true; };
  bool isConstant() const final { return false; }
  bool isSymbol() const final { return false; }
  bool isVariable() const final { return false; }
  bool isParameter() const final { return false; }

  void visit(ExprVisitor &v) const final;
  
  std::ostream &print(std::ostream &out) const final;

  bool operator==(const Expr &other) const final {
    if (!other.isNeg())
      return false;

    const auto o = static_cast<const Neg *>(&other);
    return (this->expr == o->expr);
  }

  bool operator<(const Expr &other) const final {
    if (!other.isNeg())
      return getKind() < other.getKind();

    const auto o = static_cast<const Neg *>(&other);

    if (this->expr < o->expr)
      return true;
    else
      return false;
  }
};

struct Symbol : public Expr {
  std::string n;

  bool isAffineExpr() const final { return true; }
  bool isSumExpr() const final { return false; }
  bool isMonomialExpr() const final { return true; }
  
  bool isBinOp() const final { return false; };
  bool isNeg() const final { return false; };
  bool isConstant() const final { return false; }
  bool isSymbol() const final { return true; }

  virtual void visit(ExprVisitor &v) const override;
  
protected:
  explicit Symbol(const ExprKind k, const std::string &name)
      : Expr(k), n(name) {}
};

struct Variable : public Symbol {
  explicit Variable(const std::string &name) : Symbol(EK_Variable, name) {}

  bool isConstExpr() const final { return false; }
  bool isVariable() const final { return true; }
  bool isParameter() const final { return false; }

  void visit(ExprVisitor &v) const final;
  
  std::ostream &print(std::ostream &out) const final;

  bool operator==(const Expr &other) const final {
    if (!other.isVariable())
      return false;

    const auto o = static_cast<const Variable *>(&other);
    return (this->n == o->n);
  }

  bool operator<(const Expr &other) const final {
    if (!other.isVariable())
      return getKind() < other.getKind();

    const auto o = static_cast<const Variable *>(&other);
    return (this->n < o->n);
  }
};

struct Parameter : public Symbol {
  explicit Parameter(const std::string &name) : Symbol(EK_Parameter, name) {}

  bool isConstExpr() const final { return true; }
  bool isVariable() const final { return false; }
  bool isParameter() const final { return true; }

  void visit(ExprVisitor &v) const final;
  
  std::ostream &print(std::ostream &out) const final;

  bool operator==(const Expr &other) const final {
    if (!other.isParameter())
      return false;

    const auto o = static_cast<const Parameter *>(&other);
    return (this->n == o->n);
  }

  bool operator<(const Expr &other) const final {
    if (!other.isParameter())
      return getKind() < other.getKind();

    const auto o = static_cast<const Parameter *>(&other);
    return (this->n < o->n);
  }
};

struct Constant : public Expr {
  using value_type = uint64_t;

  value_type val;

  explicit Constant(value_type value) : Expr(EK_Constant), val(value) {}

  bool isConstExpr() const final { return true; }
  bool isAffineExpr() const final { return true; }
  bool isSumExpr() const final { return false; }
  bool isMonomialExpr() const final { return true; }
  
  bool isBinOp() const final { return false; }
  bool isNeg() const final { return false; };
  bool isConstant() const final { return true; }
  bool isSymbol() const final { return false; }
  bool isVariable() const final { return false; }
  bool isParameter() const final { return false; }

  void visit(ExprVisitor &v) const final;
  
  std::ostream &print(std::ostream &out) const final;

  bool operator==(const Expr &other) const final {
    if (!other.isConstant())
      return false;

    const auto o = static_cast<const Constant *>(&other);
    return (this->val == o->val);
  }

  bool operator<(const Expr &other) const final {
    if (!other.isConstant())
      return getKind() < other.getKind();

    const auto o = static_cast<const Constant *>(&other);
    return (this->val < o->val);
  }
};

struct ExprVisitor {
  // Derived classes *must* implement the
  // visit methods for non-abstract expressions:
  virtual void visitBinOp(const BinOp *) = 0;
  virtual void visitNeg(const Neg *) = 0;
  virtual void visitConstant(const Constant *) = 0;
  virtual void visitVariable(const Variable *) = 0;
  virtual void visitParameter(const Parameter *) = 0;

  // Derived classes *may* override the following standard
  // behaviour of the visit methods for non-abstract expressions 
  virtual void visitExpr(const Expr *e) { e->visit(*this); }
  virtual void visitSymbol(const Symbol *s) { s->visit(*this); }
};

} // namespace ranges
} // namespace teckyl

#endif // TECKYL_TC_INFERENCE_EXPR_H
