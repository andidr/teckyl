#include "teckyl/tc/lang/inference/ranges.h"
#include <sstream>

namespace teckyl {
namespace ranges {

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
