#ifndef TECKYL_TC_INFERENCE_RANGES_H
#define TECKYL_TC_INFERENCE_RANGES_H

#include "teckyl/tc/lang/inference/expr.h"

#include <llvm/Support/ErrorHandling.h>
#include <memory>
#include <ostream>
#include <set>
#include <string>

namespace teckyl {
namespace ranges {

enum cmptype { LT, LE, EQ, GE, GT };

struct Constraint {
  ExprRef l;
  cmptype op;
  ExprRef r;

  explicit Constraint(const ExprRef &left, cmptype operation, const ExprRef &right)
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
  ExprRef low, up;

  explicit Range(const std::string &name, const ExprRef &lower, const ExprRef &upper)
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

  void addRange(const std::string &name, const ExprRef &lower, const ExprRef &upper) {
    const Range r(name, lower, upper);

    // avoid duplicates:
    if (solved.count(r) == 0)
      solved.insert(std::move(r));
  }

  void addConstraint(const ExprRef &left, cmptype operation, const ExprRef &right) {
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

  void addConstraints(const ExprRef &lower, const ExprRef &middle, const ExprRef &upper) {
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

#endif // TECKYL_TC_INFERENCE_RANGES_H
