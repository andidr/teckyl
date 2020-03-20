#include "teckyl/tc/lang/ranges.h"
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
  case '+':
  case '-':
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

std::ostream &operator<<(std::ostream &out, cmptype op) {
  switch (op) {
  case LT:
    return out << "<";
  case LE:
    return out << "<=";
  case EQ:
    return out << "==";
  case GE:
    return out << ">=";
  case GT:
    return out << "<";
  default:
    llvm_unreachable("Unknown comparison type");
  }
}

std::ostream &BinOp::print(std::ostream &out) const {
  return out << "(" << *l << op << *r << ")";
}

std::ostream &Variable::print(std::ostream &out) const { return out << n; }
std::ostream &Parameter::print(std::ostream &out) const {
  return out << "$" << n;
}
std::ostream &Constant::print(std::ostream &out) const { return out << val; }

std::ostream &operator<<(std::ostream &out, const Constraint &c) {
  return out << *c.l << " " << c.op << " " << *c.r;
}

std::ostream &operator<<(std::ostream &out, const Range &r) {
  return out << *r.low << " <= " << r.n << " < " << *r.up;
}

std::ostream &operator<<(std::ostream &out, const InferenceProblem &p) {
  for (const Range &r : p.solved)
    out << "Range: " << r << std::endl;

  for (const Constraint &c : p.constraints)
    out << "Constraint: " << c << std::endl;

  return out;
}

} // namespace ranges
} // namespace teckyl
