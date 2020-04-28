#ifndef TECKYL_TC_INFERENCE_TRANSFORMATION_H
#define TECKYL_TC_INFERENCE_TRANSFORMATION_H

#include "teckyl/tc/lang/inference/analysis.h"
#include "teckyl/tc/lang/inference/expr.h"

#include <algorithm>
#include <functional>
#include <llvm/Support/ErrorHandling.h>
#include <memory>
#include <string>
#include <vector>

namespace teckyl {
namespace ranges {

struct Transformation {
  virtual void reset() {}
  virtual ExprRef run(const ExprRef e) = 0;
};

struct Identity : public Transformation {
  ExprRef run(const ExprRef e) { return e; }
};

template <typename T>
struct Stack {
  std::vector<T> items;

  void push(T i) { items.push_back(i); }
  T pop() {
    auto i = items.back();
    items.pop_back();
    return i;
  }

  void clear() { items.clear(); }
  size_t size() const { return items.size(); }
};

struct StackBasedTrafo : public Transformation {
  StackBasedTrafo() { reset(); }
  virtual void reset() override { stack.clear(); }

protected:
  Stack<ExprRef> stack;
};

struct StackBasedVisitor : public StackBasedTrafo, ExprVisitor {
  ExprRef run(const ExprRef e) override {
    e->visit(*this);
    if (stack.size() != 1) {
      llvm_unreachable("Stack has been mis-managed");
    }
    return stack.pop();
  }

protected:
  // Identity visitors:

  void visitBinOp(const BinOp *b) override {
    const auto op = b->op;
    const auto left = b->l;
    const auto right = b->r;

    left->visit(*this);
    const auto left_ = stack.pop();

    right->visit(*this);
    const auto right_ = stack.pop();

    auto result = std::make_shared<BinOp>(op, left_, right_);
    stack.push(result);
  }

  void visitNeg(const Neg *n) override {
    n->expr->visit(*this);
    const auto result = std::make_shared<Neg>(stack.pop());
    stack.push(result);
  }

  void visitConstant(const Constant *c) override {
    stack.push(std::make_shared<Constant>(*c));
  }

  void visitParameter(const Parameter *p) override {
    stack.push(std::make_shared<Parameter>(*p));
  }

  void visitVariable(const Variable *v) override {
    stack.push(std::make_shared<Variable>(*v));
  }
};

struct Distribution : public StackBasedVisitor {
private:
  void visitBinOp(const BinOp *b) final {
    const auto op = b->op;
    const auto left = b->l;
    const auto right = b->r;

    left->visit(*this);
    const auto left_ = stack.pop();

    right->visit(*this);
    const auto right_ = stack.pop();

    if (op != TIMES) {
      auto result = std::make_shared<BinOp>(op, left_, right_);
      stack.push(result);
      return;
    }

    // For the remainder of this method, 'op == TIMES' holds.

    if (left_->isSumExpr()) {
      const BinOp *left__ = static_cast<const BinOp *>(left_.get());

      const auto op__ = left__->op;
      if (op__ == TIMES) {
        // We know that 'left_' is a "SumExpr":
        llvm_unreachable("Should not be here");
      }

      const ExprRef a = left__->l;
      const ExprRef b = left__->r;

      // The following holds: left__ == a 'op__' b
      // Must implement the following distribution:
      // (a 'op__' b) * right_ ~> (a * right_) 'op__' (b * right_)

      const auto a_ = std::make_shared<BinOp>(TIMES, a, right_);
      a_->visit(*this);
      const auto a__ = stack.pop();

      const auto b_ = std::make_shared<BinOp>(TIMES, b, right_);
      b_->visit(*this);
      const auto b__ = stack.pop();

      const auto result = std::make_shared<BinOp>(op__, a__, b__);
      stack.push(result);
      return;
    }

    if (right_->isSumExpr()) {
      const BinOp *right__ = static_cast<const BinOp *>(right_.get());

      const auto op__ = right__->op;
      if (op__ == TIMES) {
        // We know that 'right_' is a "SumExpr":
        llvm_unreachable("Should not be here");
      }
      const ExprRef a = right__->l;
      const ExprRef b = right__->r;

      // The following holds: right__ == a 'op__' b
      // Must implement the following distribution:
      // left_ * (a 'op__' b) ~> (left_ * a) 'op__' (left_ * b)

      const auto a_ = std::make_shared<BinOp>(TIMES, left_, a);
      a_->visit(*this);
      const auto a__ = stack.pop();

      const auto b_ = std::make_shared<BinOp>(TIMES, left_, b);
      b_->visit(*this);
      const auto b__ = stack.pop();

      const auto result = std::make_shared<BinOp>(op__, a__, b__);
      stack.push(result);
      return;
    }

    auto result = std::make_shared<BinOp>(TIMES, left_, right_);
    stack.push(result);
  }
};

struct SignConversion : public StackBasedVisitor {
  // Convert all signs, i.e. 'Neg' expressions and 'MINUS' operators,
  // by moving them deeper into expressions until the only signs appear
  // as 'Neg' expressions around variables, parameters or constants.

  SignConversion() { reset(); }

  void reset() override {
    StackBasedVisitor::reset();
    collectedSigns = 0;
  }

private:
  unsigned collectedSigns;

  void visitBinOp(const BinOp *b) final {
    auto op = b->op;
    const auto left = b->l;
    const auto right = b->r;

    left->visit(*this);
    ExprRef left_ = stack.pop();

    ExprRef right_;

    switch (op) {
    case TIMES: {
      // Pass signs only down the left argument of a multiplication
      // (and not down the right argument):
      unsigned savedSigns = collectedSigns;
      collectedSigns = 0;

      right->visit(*this);
      right_ = stack.pop();

      collectedSigns = savedSigns;
      break;
    }
    case MINUS: {
      // An extra sign is passed down the right
      // argument of a subtraction:
      ++collectedSigns;

      right->visit(*this);
      right_ = stack.pop();

      --collectedSigns;
      op = PLUS;
      break;
    }
    case PLUS: {
      right->visit(*this);
      right_ = stack.pop();
      break;
    }
    default:
      llvm_unreachable("Invalid operator");
    }

    const auto result = std::make_shared<BinOp>(op, left_, right_);
    stack.push(result);
    return;
  }

  void visitNeg(const Neg *n) final {
    ++collectedSigns;
    n->expr->visit(*this);
    --collectedSigns;
    // Leave result of the last call to 'visit' on the stack.
  }

  void push_with_sign(ExprRef e) {
    if (collectedSigns & 1)
      stack.push(std::make_shared<Neg>(e));
    else
      stack.push(e);
  }

  void visitConstant(const Constant *c) final {
    push_with_sign(std::make_shared<Constant>(*c));
  }

  void visitParameter(const Parameter *p) final {
    push_with_sign(std::make_shared<Parameter>(*p));
  }

  void visitVariable(const Variable *v) final {
    push_with_sign(std::make_shared<Variable>(*v));
  }
};

struct Normalization : public Transformation {
  Normalization(bool leftAssociate = true) : leftAssoc(leftAssociate) {}

  Normalization() { reset(); }

  void reset() final {
    D.reset();
    SC.reset();
    MC.reset();
  }

  using MonomialVector = std::vector<Monomial>;

  ExprRef run(const ExprRef e) final { return runImpl(e); }

private:
  bool leftAssoc;

  Distribution D;
  SignConversion SC;

  MonomialCollection MC;

  ExprRef runImpl(const ExprRef e0) {
    const auto e1 = SC.run(e0);
    const auto e2 = D.run(e1);

    MC.run(e2);
    MonomialVector monos = MC.get();

    const ExprRef result = leftAssoc ? toExprL(monos) : toExprR(monos);
    return result;
  }

  static ExprRef toExprL(const MonomialVector &monos) {
    if (monos.size() == 0)
      llvm_unreachable("Expression must have at least one monomial");

    ExprRef expr = monos[0].toExprL();
    for (auto m = monos.begin() + 1; m != monos.end(); m++) {
      expr = std::make_shared<BinOp>(PLUS, expr, m->toExprL());
    }

    return expr;
  }

  static ExprRef toExprR(const MonomialVector &monos) {
    if (monos.size() == 0)
      llvm_unreachable("Expression must have at least one monomial");

    MonomialVector ms = monos;
    std::reverse(ms.begin(), ms.end());

    ExprRef expr = ms[0].toExprR();
    for (auto m = ms.begin() + 1; m != ms.end(); m++) {
      expr = std::make_shared<BinOp>(PLUS, m->toExprR(), expr);
    }

    return expr;
  }
};

struct Substitution : public StackBasedVisitor {
  using Assignment =
      std::function<ExprRef(const std::string &, const ExprRef &)>;

  static const Assignment identity;

  Substitution(const Assignment &variablesAssignment = identity,
               const Assignment &parametersAssignment = identity)
      : varsSubst(variablesAssignment), paramsSubst(parametersAssignment) {}

private:
  Assignment varsSubst;
  Assignment paramsSubst;

  void visitVariable(const Variable *v) {
    stack.push(varsSubst(v->n, std::make_shared<Variable>(*v)));
  }

  void visitParameter(const Parameter *p) {
    stack.push(paramsSubst(p->n, std::make_shared<Parameter>(*p)));
  }
};

struct SingleSubstitution : public Substitution {
  enum SubstitutionTarget { Variable, Parameter };

  SingleSubstitution(const std::string &targetName,
                     const ExprRef &exprToSubstitute,
                     SubstitutionTarget targetKind = Variable)
      : Substitution(targetKind == Variable ? nameSubstitution : identity,
                     targetKind == Parameter ? nameSubstitution : identity),
        name(targetName), expr(exprToSubstitute) {}

private:
  const Assignment nameSubstitution = [this](const std::string &name,
                                             const ExprRef &self) {
    return (name == this->name) ? this->expr : self;
  };

  std::string name;
  const ExprRef expr;
};
} // namespace ranges
} // namespace teckyl

#endif // TECKYL_TC_INFERENCE_TRANSFORMATION_H
