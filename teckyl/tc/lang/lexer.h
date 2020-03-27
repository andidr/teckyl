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
#ifndef TECKYL_TC_LANG_LEXER_H_
#define TECKYL_TC_LANG_LEXER_H_

#include <algorithm>
#include <assert.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace lang {

// single character tokens are just the character itself '+'
// multi-character tokens need an entry here
// if the third entry is not the empty string, it is used
// in the lexer to match this token.

// These kinds are also used in Tree.h as the kind of the AST node.
// Some kinds TK_APPLY, TK_LIST are only used in the AST and are not seen in the
// lexer.

#define TC_FORALL_TOKEN_KINDS(_)                                               \
  _(TK_EOF, "eof", "")                                                         \
  _(TK_NUMBER, "number", "")                                                   \
  _(TK_BOOL_VALUE, "bool_value", "")                                           \
  _(TK_MIN, "min", "min")                                                      \
  _(TK_MAX, "max", "max")                                                      \
  _(TK_WHERE, "where", "where")                                                \
  _(TK_DEF, "def", "def")                                                      \
  _(TK_ARROW, "arrow", "->")                                                   \
  _(TK_EQUIVALENT, "equivalent", "<=>")                                        \
  _(TK_IDENT, "ident", "")                                                     \
  _(TK_STRING, "string", "")                                                   \
  _(TK_CONST, "const", "")                                                     \
  _(TK_LIST, "list", "")                                                       \
  _(TK_OPTION, "option", "")                                                   \
  _(TK_APPLY, "apply", "")                                                     \
  _(TK_COMPREHENSION, "comprehension", "")                                     \
  _(TK_TENSOR_TYPE, "tensor_type", "")                                         \
  _(TK_RANGE_CONSTRAINT, "range_constraint", "")                               \
  _(TK_PARAM, "param", "")                                                     \
  _(TK_INFERRED, "inferred", "")                                               \
  _(TK_ACCESS, "access", "")                                                   \
  _(TK_BUILT_IN, "built-in", "")                                               \
  _(TK_PLUS_EQ, "plus_eq", "+=")                                               \
  _(TK_TIMES_EQ, "times_eq", "*=")                                             \
  _(TK_MIN_EQ, "min_eq", "min=")                                               \
  _(TK_MAX_EQ, "max_eq", "max=")                                               \
  _(TK_PLUS_EQ_B, "plus_eq_b", "+=!")                                          \
  _(TK_TIMES_EQ_B, "times_eq_b", "*=!")                                        \
  _(TK_MIN_EQ_B, "min_eq_b", "min=!")                                          \
  _(TK_MAX_EQ_B, "max_eq_b", "max=!")                                          \
                                                                               \
  _(TK_BOOL, "bool", "bool")                                                   \
  _(TK_UINT8, "uint8", "uint8")                                                \
  _(TK_UINT16, "uint16", "uint16")                                             \
  _(TK_UINT32, "uint32", "uint32")                                             \
  _(TK_UINT64, "uint64", "uint64")                                             \
  _(TK_INT8, "int8", "int8")                                                   \
  _(TK_INT16, "int16", "int16")                                                \
  _(TK_INT32, "int32", "int32")                                                \
  _(TK_INT64, "int64", "int64")                                                \
  _(TK_SIZET, "size_t", "size_t")                                              \
  _(TK_FLOAT16, "float16", "float16")                                          \
  _(TK_FLOAT32, "float32", "float32")                                          \
  _(TK_FLOAT64, "float64", "float64")                                          \
  _(TK_FLOAT, "float", "float")                                                \
  _(TK_DOUBLE, "double", "double")                                             \
  _(TK_CAST, "cast", "")                                                       \
  _(TK_IN, "in", "in")                                                         \
  _(TK_GE, "ge", ">=")                                                         \
  _(TK_LE, "le", "<=")                                                         \
  _(TK_EQ, "eq", "==")                                                         \
  _(TK_NE, "neq", "!=")                                                        \
  _(TK_AND, "and", "&&")                                                       \
  _(TK_OR, "or", "||")                                                         \
  _(TK_LET, "let", "")                                                         \
  _(TK_EXISTS, "exists", "exists")

static const char *valid_single_char_tokens = "+-*/()[]?:,={}><!%";

enum TokenKind {
  // we use characters to represent themselves so skip all valid characters
  // before
  // assigning enum values to multi-char tokens.
  TK_DUMMY_START = 256,
#define DEFINE_TOKEN(tok, _, _2) tok,
  TC_FORALL_TOKEN_KINDS(DEFINE_TOKEN)
#undef DEFINE_TOKEN
};

// Returns a human-readable description of the token
std::string kindToString(int kind);
// Returns the string used by the lexer to match a given token, or throws
// if it can't be produced by the lexer.
std::string kindToToken(int kind);

// nested hash tables that indicate char-by-char what is a valid token.
struct TokenTrie;
using TokenTrieRef = std::unique_ptr<TokenTrie>;
struct TokenTrie {
  TokenTrie() : kind(0) {}
  void insert(const char *str, int tok) {
    if (*str == '\0') {
      assert(kind == 0);
      kind = tok;
      return;
    }
    auto &entry = children[*str];
    if (entry == nullptr) {
      entry.reset(new TokenTrie());
    }
    entry->insert(str + 1, tok);
  }
  int kind; // 0 == invalid token
  std::unordered_map<char, TokenTrieRef> children;
};

// stuff that is shared against all TC lexers/parsers and is initialized only
// once.
struct SharedParserData {
  SharedParserData() : head(new TokenTrie()) {
    // listed in increasing order of precedence
    std::vector<std::vector<int>> binary_ops = {
        {'?'},      {TK_OR},
        {TK_AND},   {'>', '<', TK_LE, TK_GE, TK_EQ, TK_NE},
        {'+', '-'}, {'*', '/', '%'},
    };
    std::vector<std::vector<int>> unary_ops = {
        {'-', '!'},
    };

    std::stringstream ss;
    for (const char *c = valid_single_char_tokens; *c; c++) {
      const char str[] = {*c, '\0'};
      head->insert(str, *c);
    }

#define ADD_CASE(tok, _, tokstring)                                            \
  if (*tokstring != '\0') {                                                    \
    head->insert(tokstring, tok);                                              \
  }
    TC_FORALL_TOKEN_KINDS(ADD_CASE)
#undef ADD_CASE

    // precedence starts at 1 so that there is always a 0 precedence
    // less than any other precedence
    int prec = 1;
    for (auto &group : binary_ops) {
      for (auto &element : group) {
        binary_prec[element] = prec;
      }
      prec++;
    }
    // unary ops
    for (auto &group : unary_ops) {
      for (auto &element : group) {
        unary_prec[element] = prec;
      }
      prec++;
    }
  }
  bool isNumber(const std::string &str, size_t start, size_t *len) {
    char first = str[start];
    // strtod allows numbers to start with + or -
    // http://en.cppreference.com/w/cpp/string/byte/strtof
    // but we want only the number part, otherwise 1+3 will turn into two
    // adjacent numbers in the lexer
    if (first == '-' || first == '+')
      return false;
    const char *startptr = str.c_str() + start;
    char *endptr;
    std::strtod(startptr, &endptr);
    *len = endptr - startptr;
    if (*len == 0)
      return false;

    bool isFloatLiteral = false;

    for (const char *tokptr = startptr; tokptr != endptr; tokptr++) {
      if (*tokptr == '.' || *tokptr == 'e') {
        isFloatLiteral = true;
        break;
      }
    }

    // It's safe to dereference endptr, since as per the specification
    // of strtod, it is either equal to startptr or the address of the
    // character past startptr. Since startptr is initialized with
    // std::string::c_str(), it is guaranteed to point to a sequence
    // of characters terminated by zero, so endptr points at most at
    // the NUL character at the end of the string.
    //
    // Similarly, the use of C string functions are safe here, since
    // the above check guarantees that endptr hasn't moved past the
    // NUL character.
    if (*endptr != '\0') {
      static const char *suffixes[] = {"i8", "i16", "i32", "i64",
                                       "u8", "u16", "u32", "u64",
                                       "z",  "f16", "f32", "f64"};

#define ARRAY_SIZE(a) ((sizeof(a) / sizeof(a[0])))

      for (size_t i = 0; i < ARRAY_SIZE(suffixes); i++) {
        size_t sufflen = strlen(suffixes[i]);

        if (std::strncmp(endptr, suffixes[i], sufflen) == 0) {
          *len += sufflen;

          // Float literals must have a float type suffix
          if (isFloatLiteral && suffixes[i][0] != 'f')
            return false;
          else
            return true;
        }
      }
    }

    // Constant without type suffix
    return true;
  }
  // find the longest match of str.substring(pos) against a token, return true
  // if successful
  // filling in kind, start,and len
  bool match(const std::string &str, size_t pos, int *kind, size_t *start,
             size_t *len, size_t *line, size_t *ch) {
    // skip whitespace
    while (pos < str.size() && isspace(str[pos])) {
      if (str[pos] == '\n') {
        (*line)++;
        *ch = 1;
      } else {
        (*ch)++;
      }

      pos++;
    }
    // skip comments
    if (pos < str.size() && str[pos] == '#') {
      while (pos < str.size() && str[pos] != '\n') {
        pos++;
        (*ch)++;
      }

      if (pos < str.size() && str[pos] == '\n') {
        *ch = 1;
        (*line)++;
      }

      // tail call, handle whitespace and more comments
      return match(str, pos, kind, start, len, line, ch);
    }
    *start = pos;
    if (pos == str.size()) {
      *kind = TK_EOF;
      *len = 0;
      return true;
    }
    // check for a valid number
    if (isNumber(str, pos, len)) {
      *kind = TK_NUMBER;
      return true;
    }
    // check for either an ident or a token
    // ident tracks whether what we have scanned so far could be an identifier
    // matched indicates if we have found any match.
    bool matched = false;
    bool ident = true;
    TokenTrie *cur = head.get();
    for (size_t i = 0; pos + i < str.size() && (ident || cur != nullptr); i++) {
      ident = ident && validIdent(i, str[pos + i]);
      if (ident) {
        matched = true;
        *len = i + 1;
        *kind = TK_IDENT;
      }
      // check for token second, so that e.g. 'max' matches the token TK_MAX
      // rather the
      // identifier 'max'
      if (cur) {
        auto it = cur->children.find(str[pos + i]);
        cur = (it == cur->children.end()) ? nullptr : it->second.get();
        if (cur && cur->kind != 0) {
          matched = true;
          *len = i + 1;
          *kind = cur->kind;
        }
      }
    }
    return matched;
  }
  bool isUnary(int kind, int *prec) {
    auto it = unary_prec.find(kind);
    if (it != unary_prec.end()) {
      *prec = it->second;
      return true;
    }
    return false;
  }
  bool isBinary(int kind, int *prec) {
    auto it = binary_prec.find(kind);
    if (it != binary_prec.end()) {
      *prec = it->second;
      return true;
    }
    return false;
  }
  bool isRightAssociative(int kind) {
    switch (kind) {
    case '?':
      return true;
    default:
      return false;
    }
  }
  bool isScalarType(int kind) {
    switch (kind) {
    case TK_BOOL:
    case TK_UINT8:
    case TK_UINT16:
    case TK_UINT32:
    case TK_UINT64:
    case TK_INT8:
    case TK_INT16:
    case TK_INT32:
    case TK_INT64:
    case TK_SIZET:
    case TK_FLOAT16:
    case TK_FLOAT32:
    case TK_FLOAT64:
    case TK_FLOAT:
    case TK_DOUBLE:
      return true;
    default:
      return false;
    }
  }

private:
  bool validIdent(size_t i, char n) {
    return isalpha(n) || n == '_' || (i > 0 && isdigit(n));
  }
  TokenTrieRef head;
  std::unordered_map<int, int>
      unary_prec; // map from token to its unary precedence
  std::unordered_map<int, int>
      binary_prec; // map from token to its binary precedence
};

SharedParserData &sharedParserData();

// a range of a shared string 'source_' with functions to help debug by highlight
// that
// range.
struct SourceRange {
  SourceRange(const std::shared_ptr<std::string> &source_,
              const std::shared_ptr<std::string> &filename_, size_t start_,
              size_t end_, size_t start_line_, size_t start_ch_,
              size_t end_line_, size_t end_ch_)
      : source_(source_), filename_(filename_), start_(start_), end_(end_),
        start_line_(start_line_), start_ch_(start_ch_), end_line_(end_line_),
        end_ch_(end_ch_) {}

  const std::string text() const {
    return source().substr(start(), end() - start());
  }
  size_t size() const { return end() - start(); }
  void highlight(std::ostream &out) const {
    const std::string &str = source();
    size_t begin = start();
    size_t end = start();
    while (begin > 0 && str[begin - 1] != '\n')
      --begin;
    while (end < str.size() && str[end] != '\n')
      ++end;
    out << str.substr(0, end) << "\n";
    out << std::string(start() - begin, ' ');
    size_t len = std::min(size(), end - start());
    out << std::string(len, '~')
        << (len < size() ? "...  <--- HERE" : " <--- HERE");
    out << str.substr(end);
    if (str.size() > 0 && str.back() != '\n')
      out << "\n";
  }
  const std::string &source() const { return *source_; }
  const std::shared_ptr<std::string> &source_ptr() const { return source_; }
  const std::string &filename() const { return *filename_; }
  const std::shared_ptr<std::string> &filename_ptr() const { return filename_; }
  size_t start() const { return start_; }
  size_t end() const { return end_; }

  size_t startLine() const { return start_line_; }
  size_t endLine() const { return end_line_; }

  size_t startCharacter() const { return start_ch_; }
  size_t endCharacter() const { return end_ch_; }

private:
  std::shared_ptr<std::string> source_;
  std::shared_ptr<std::string> filename_;
  size_t start_;
  size_t end_;
  size_t start_line_;
  size_t start_ch_;
  size_t end_line_;
  size_t end_ch_;
};

struct Token {
  int kind;
  SourceRange range;
  Token(int kind, const SourceRange &range) : kind(kind), range(range) {}

  // Returns the numerical portion of the string without suffix for
  // TK_NUMBER
  std::string numStringValue() {
    assert(TK_NUMBER == kind);
    size_t idx;
    std::stod(text(), &idx);
    assert(idx > 0);

    if (idx < range.size()) {
      std::string suffix = text().substr(idx);

      assert(suffix == "f16" || suffix == "f32" || suffix == "f64" ||
             suffix == "u8" || suffix == "u16" || suffix == "u32" ||
             suffix == "u64" || suffix == "i8" || suffix == "i16" ||
             suffix == "i32" || suffix == "i64" || suffix == "z");
    } else {
      assert(idx == range.size());
    }

    return text().substr(0, idx);
  }
  // Returns the suffix for the number literal (either "u8", "u16",
  // "u32", "u64", "i8", "i16", "i32", "i64", "f16", "f32", "f64", "z"
  // or the empty string "" if no suffix has been specified
  // originally.
  std::string numSuffix() {
    assert(TK_NUMBER == kind);
    size_t idx;
    std::stod(text(), &idx);

    return text().substr(idx);
  }
  std::string text() { return range.text(); }
  std::string kindString() const { return kindToString(kind); }
};

struct Lexer {
  std::shared_ptr<std::string> source;
  std::shared_ptr<std::string> filename;

  Lexer(const std::string &source_,
        const std::string &filename_ = "(unknown file)")
      : source(std::make_shared<std::string>(source_)),
        filename(std::make_shared<std::string>(filename_)), pos(0), line(1),
        ch(1), cur_(TK_EOF, SourceRange(source, filename, 0, 0, 0, 0, 0, 0)),
        shared(sharedParserData()) {
    next();
  }
  bool nextIf(int kind) {
    if (cur_.kind != kind)
      return false;
    next();
    return true;
  }
  Token lookahead() {
    if (!lookahead_) {
      lookahead_.reset(new Token(lex()));
    }
    return *lookahead_;
  }
  Token next() {
    auto r = cur_;
    if (lookahead_) {
      cur_ = *lookahead_;
      lookahead_.reset();
    } else {
      cur_ = lex();
    }
    return r;
  }
  void reportError(const std::string &what, const Token &t);
  void reportError(const std::string &what) { reportError(what, cur_); }
  Token expect(int kind) {
    if (cur_.kind != kind) {
      reportError(kindToString(kind));
    }
    return next();
  }
  Token &cur() { return cur_; }

private:
  Token lex() {
    int kind = -1;
    size_t start = 0;
    size_t length = 0;
    size_t start_line = line;
    size_t start_ch = ch;

    assert(source);
    if (!shared.match(*source, pos, &kind, &start, &length, &line, &ch)) {
      reportError(
          "a valid token",
          Token((*source)[start],
                SourceRange(source, filename, start, start + 1, start_line,
                            start_ch, start_line, start_ch)));
    }
    auto t = Token(kind, SourceRange(source, filename, start, start + length,
                                     start_line, start_ch, line, ch));
    pos = start + length;
    return t;
  }
  size_t pos;
  size_t line;
  size_t ch;
  Token cur_;
  std::unique_ptr<Token> lookahead_;
  SharedParserData &shared;
};
} // namespace lang

#endif // TECKYL_TC_LANG_LEXER_H_
