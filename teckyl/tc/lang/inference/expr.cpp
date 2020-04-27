#include "teckyl/tc/lang/inference/expr.h"
#include <sstream>

namespace teckyl {
namespace ranges {

// Creates an optype from a TC language kind
static enum optype opFromLangKind(int kind) {
  switch (kind) {
  case '+':
    return optype::PLUS;
  case '-':
    return optype::MINUS;
  case '*':
    return optype::TIMES;
  default:
    llvm_unreachable("Unknown kind");
  }
}

// Creates an expression from a TC expression. Fails with an fatal
// error if there is no equivalent expression type for the TC
// expression type.
ExprRef Expr::fromTreeRef(const lang::TreeRef &t,
                          const std::unordered_set<std::string> &rangeParams) {
  switch (t->kind()) {
  case '-': {
    if (t->trees().size() == 1) // Unary negation
      return std::make_shared<Neg>(fromTreeRef(t->trees()[0], rangeParams));
    else if (t->trees().size() == 2) // Binary minus
      return std::make_shared<BinOp>(opFromLangKind(t->kind()),
                                     fromTreeRef(t->trees()[0], rangeParams),
                                     fromTreeRef(t->trees()[1], rangeParams));
    else
      llvm_unreachable("Invalid number of operands");
  }

  case '+':
  case '*':
    return std::make_shared<BinOp>(opFromLangKind(t->kind()),
                                   fromTreeRef(t->trees()[0], rangeParams),
                                   fromTreeRef(t->trees()[1], rangeParams));
  case lang::TK_IDENT: {
    std::string name = lang::Ident(t).name();

    if (rangeParams.count(name))
      return std::make_shared<Parameter>(name);
    else
      return std::make_shared<Variable>(name);
  }

  case lang::TK_CONST:
    return std::make_shared<Constant>(
        lang::Const(t).value<Constant::value_type>());
  }

  llvm_unreachable("Unknown kind");
}

void Expr::visit(ExprVisitor &v) const {
  v.visitExpr(this);
}
  
std::ostream &operator<<(std::ostream &out, optype op) {
  switch (op) {
  case PLUS:
    return out << "+";
  case MINUS:
    return out << "-";
  case TIMES:
    return out << "*";
  default:
    llvm_unreachable("Unknown op");
  }
}

std::ostream &BinOp::print(std::ostream &out) const {
  return out << "(" << *l << op << *r << ")";
}

void BinOp::visit(ExprVisitor &v) const {
  v.visitBinOp(this);
}

std::ostream &Neg::print(std::ostream &out) const {
  return out << "(-" << *expr << ")";
}

void Neg::visit(ExprVisitor &v) const {
  v.visitNeg(this);
}

void Symbol::visit(ExprVisitor &v) const {
  v.visitSymbol(this);
}
  
std::ostream &Variable::print(std::ostream &out) const {
  return out << n;
}

void Variable::visit(ExprVisitor &v) const {
  v.visitVariable(this);
}
  
std::ostream &Parameter::print(std::ostream &out) const {
  return out << "$" << n;
}

void Parameter::visit(ExprVisitor &v) const {
  v.visitParameter(this);
}

std::ostream &Constant::print(std::ostream &out) const { return out << val; }

void Constant::visit(ExprVisitor &v) const {
  v.visitConstant(this);
}

} // namespace ranges
} // namespace teckyl
