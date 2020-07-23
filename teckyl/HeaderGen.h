#ifndef TECKYL_HEADER_GEN_H
#define TECKYL_HEADER_GEN_H

#include <map>
#include <string>

#include "teckyl/tc/lang/tree_views.h"

namespace teckyl {
std::string genHeader(const std::map<std::string, lang::Def> &tcs,
                      const std::string &includeGuard);
}

#endif
