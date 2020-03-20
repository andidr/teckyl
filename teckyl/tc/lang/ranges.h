#ifndef TECKYL_TC_RANGES_H
#define TECKYL_TC_RANGES_H

#include "teckyl/tc/lang/tree_views.h"

#include <llvm/Support/ErrorHandling.h>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <unordered_set>

namespace teckyl {
namespace ranges {

// Kinds of (non-abstract) expressions,
// used for comparison operator '<':
enum ExprKind { EK_BinOp, EK_Variable, EK_Parameter, EK_Constant };

struct Expr;
using ExprRef = std::shared_ptr<Expr>;

// Base class for expressions for range inference
struct Expr {
  explicit Expr(ExprKind k) : kind(k) {}

  virtual bool isConstExpr() const = 0;
  virtual bool isAffineExpr() const = 0;
  virtual bool isBinOp() const = 0;
  virtual bool isConstant() const = 0;
  virtual bool isSymbol() const = 0;
  virtual bool isVariable() const = 0;
  virtual bool isParameter() const = 0;

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

  bool isBinOp() const final { return true; };
  bool isConstant() const final { return false; }
  bool isSymbol() const final { return false; }
  bool isVariable() const final { return false; }
  bool isParameter() const final { return false; }

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

struct Symbol : public Expr {
  std::string n;

  bool isAffineExpr() const final { return true; }
  bool isBinOp() const final { return false; };
  bool isConstant() const final { return false; }
  bool isSymbol() const final { return true; }

protected:
  explicit Symbol(const ExprKind k, const std::string &name)
      : Expr(k), n(name) {}
};

struct Variable : public Symbol {
  explicit Variable(const std::string &name) : Symbol(EK_Variable, name) {}

  bool isConstExpr() const final { return false; }
  bool isVariable() const final { return true; }
  bool isParameter() const final { return false; }

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
  bool isBinOp() const final { return false; }
  bool isConstant() const final { return true; }
  bool isSymbol() const final { return false; }
  bool isVariable() const final { return false; }
  bool isParameter() const final { return false; }

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

enum cmptype { LT, LE, EQ, GE, GT };

struct Constraint {
  std::shared_ptr<Expr> l;
  cmptype op;
  std::shared_ptr<Expr> r;

  explicit Constraint(std::shared_ptr<Expr> left, cmptype operation,
                      std::shared_ptr<Expr> right)
      : l(left), op(operation), r(right) {}

  explicit Constraint(const Constraint &other)
      : l(other.l), op(other.op), r(other.r) {}

  ~Constraint() {
    l.reset();
    r.reset();
  }

  const Constraint &operator=(const Constraint &other) {
    l.reset(other.l.get());
    op = other.op;
    r.reset(other.r.get());
    return *this;
  }

  explicit Constraint(Constraint &&other)
      : l(std::move(other.l)), op(other.op), r(std::move(other.r)) {}

  const Constraint &operator=(Constraint &&other) {
    l = std::move(other.l);
    op = other.op;
    r = std::move(other.r);
    return *this;
  }

  bool operator==(const Constraint &other) const {
    return (*l.get()) == (*other.l.get()) && op == other.op &&
           (*r.get()) == (*other.r.get());
  }

  bool operator<(const Constraint &other) const {
    if (l->operator<(*other.l)) {
      return true;
    } else if (l->operator==(*other.l) && op < other.op) {
      return true;
    } else if (l->operator==(*other.l) && op == other.op &&
               r->operator<(*other.r)) {
      return true;
    } else {
      return false;
    }
  }
};

using ConstraintSet = std::set<Constraint>;

// A 'Range' represents two constraints:
//  (1) 'lower' LE 'name'
//  (2) 'name'  LT 'upper'
// These constraints can be considered solved since they specify
// an explicit range for the variable 'name'.
struct Range {
  std::string n;
  std::shared_ptr<Expr> low, up;

  explicit Range(const std::string &name, const std::shared_ptr<Expr> &lower,
                 const std::shared_ptr<Expr> &upper)
      : n(name), low(lower), up(upper) {}

  ConstraintSet asConstraints() const {
    ConstraintSet res;
    auto var = std::make_shared<Variable>(n);
    res.emplace(low, LE, var);
    res.emplace(var, LT, up);
    return res;
  }

  explicit Range(const Range &other)
      : n(other.n), low(other.low), up(other.up) {}

  ~Range() {
    low.reset();
    up.reset();
  }

  const Range &operator=(const Range &other) {
    n = other.n;
    low.reset(other.low.get());
    up.reset(other.up.get());
    return *this;
  }

  explicit Range(Range &&other)
      : n(other.n), low(std::move(other.low)), up(std::move(other.up)) {}

  const Range &operator=(Range &&other) {
    n = other.n;
    low = std::move(other.low);
    up = std::move(other.up);
    return *this;
  }

  bool operator==(const Range &other) const {
    return n == other.n && (*low.get()) == (*other.low.get()) &&
           (*up.get()) == (*other.up.get());
  }

  bool operator<(const Range &other) const {
    if (n < other.n) {
      return true;
    } else if (n == other.n && low->operator<(*other.low)) {
      return true;
    } else if (n == other.n && low->operator==(*other.low) &&
               up->operator<(*other.up)) {
      return true;
    } else {
      return false;
    }
  }
};

using RangeSet = std::set<Range>;

struct InferenceProblem {
  RangeSet solved;
  ConstraintSet constraints;

  void addRange(const std::string &name, const std::shared_ptr<Expr> &lower,
                const std::shared_ptr<Expr> &upper) {
    const Range r(name, lower, upper);

    // avoid duplicates:
    if (solved.count(r) == 0)
      solved.insert(std::move(r));
  }

  void addConstraint(const std::shared_ptr<Expr> &left, cmptype operation,
                     const std::shared_ptr<Expr> &right) {
    const Constraint c(left, operation, right);

    // avoid duplicates:
    for (const auto &r : solved) {
      if (r.asConstraints().count(c))
        return;
    }

    // avoid duplicates:
    if (constraints.count(c) == 0)
      constraints.insert(std::move(c));
  }

  void addConstraints(const std::shared_ptr<Expr> &lower,
                      const std::shared_ptr<Expr> &middle,
                      const std::shared_ptr<Expr> &upper) {
    if (lower->isConstExpr() && middle->isVariable() && upper->isConstExpr()) {
      const auto var = static_cast<Variable *>(middle.get());
      addRange(var->n, lower, upper);

      // avoid duplicates:
      constraints.erase(Constraint(lower, LE, middle));
      constraints.erase(Constraint(middle, LT, upper));
    } else {
      addConstraint(lower, LE, middle);
      addConstraint(middle, LT, upper);
    }
  }

  void clear() {
    solved.clear();
    constraints.clear();
  }
};

std::ostream &operator<<(std::ostream &out, const InferenceProblem &p);

} // namespace ranges
} // namespace teckyl

#endif // TECKYL_TC_RANGES_H
