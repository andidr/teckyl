#include "teckyl/tc/lang/inference/transformation.h"
#include <sstream>

namespace teckyl {
namespace ranges {

const Substitution::Assignment Substitution::identity =
    [](const std::string &name, const ExprRef &self) { return self; };

} // namespace ranges
} // namespace teckyl
