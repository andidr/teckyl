/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tc/lang/lexer.h"

#include <cstring>

#include "tc/lang/error_report.h"

namespace lang {

std::string kindToString(int kind) {
  if (kind < 256)
    return std::string(1, kind);
  switch (kind) {
#define DEFINE_CASE(tok, str, _) \
  case tok:                      \
    return str;
    TC_FORALL_TOKEN_KINDS(DEFINE_CASE)
#undef DEFINE_CASE
    default:
      ;
  }
  std::runtime_error err("unknown kind: " + std::to_string(kind));
  THROW_OR_ASSERT(err);
}

std::string kindToToken(int kind) {
  if (kind < 256)
    return std::string(1, kind);
  switch (kind) {
#define DEFINE_CASE(tok, _, str)                                       \
  case tok:                                                            \
    if (!strcmp(str, "")) {                                            \
      std::runtime_error err("No token for: " + kindToString(kind));   \
      THROW_OR_ASSERT(err);                                            \
    }                                                                  \
    return str;
    TC_FORALL_TOKEN_KINDS(DEFINE_CASE)
#undef DEFINE_CASE
    default:
      ;
  }
  std::runtime_error err("unknown kind: " + std::to_string(kind));
  THROW_OR_ASSERT(err);
}

SharedParserData& sharedParserData() {
  static SharedParserData data; // safely handles multi-threaded init
  return data;
}

void Lexer::reportError(const std::string& what, const Token& t) {
  ErrorReport err(t.range);
  err << "expected " << what << " but found '" << t.kindString() << "' here:";
  THROW_OR_ASSERT(err);
}

} // namespace lang
