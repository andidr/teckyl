#ifndef TECKYL_TC_INFERENCE_EXPRESSION_PARSER_H_
#define TECKYL_TC_INFERENCE_EXPRESSION_PARSER_H_

#include "teckyl/tc/lang/inference/expr.h"

#include <cctype>
#include <llvm/Support/ErrorHandling.h>
#include <string>

// This file implements a simple parser for arithmetic expressions that
// are allowed to appear in range constraints. The parser is intended
// to be used for testing analyses and transformations of arithmetic
// expressions.
//
// The parser is based on the following grammar:
//
// expr -> term
//
// term -> product terms
//
// terms -> ('+' | '-') product terms
// terms -> "empty"
//
// product -> atom products
//
// products -> '*' atom products
// products -> "empty"
//
// atom -> negation
//       | EXPR_TK_VARIABLE
//       | EXPR_TK_PARAMETER
//       | EXPR_TK_CONSTANT
//       | '(' term ')'
//
// negation -> '-' atom
//
// Note that this grammar leads to parse trees in which the operations
// '+', '-' and '*' are associated to the left.
//
// Comments start with a '#' and continue until the end of the line.
//

namespace teckyl {
namespace ranges {

enum TokenKind {
  EXPR_TK_CONSTANT,  // integer literal
  EXPR_TK_VARIABLE,  // identifier, beginning with a letter
  EXPR_TK_PARAMETER, // identifier, beginning with '$' and a letter
  EXPR_TK_TIMES,     // '*'
  EXPR_TK_MINUS,     // '-'
  EXPR_TK_PLUS,      // '+'
  EXPR_TK_LPAREN,    // '('
  EXPR_TK_RPAREN     // ')'
};

struct Token {
  TokenKind kind;
  std::string lexeme;

  size_t start;
  size_t end;
};

struct ExprLexer {
  explicit ExprLexer(const std::string &source)
      : input(source), pos(0), eof(input.size() == 0) {}

  Token getCurrent() const { return current; }
  bool atEOF() const { return eof; }

  void next() { lex(); }

  void expect(TokenKind k) {
    if (k != current.kind)
      llvm_unreachable("Unexpected token");
  }

  void consumeExpected(TokenKind k) {
    expect(k);
    next();
  }

private:
  std::string input;
  size_t pos;

  bool eof;

  Token current;

  void skipSpace() {
    while (pos < input.size() && isspace(input[pos])) {
      ++pos;
    }
  }

  bool testAndSkipComment() {
    if (pos == input.size())
      return false;

    if (input[pos] != '#')
      return false;

    // Skip comment until the end of the line:
    while (pos < input.size() && input[pos] != '\n') {
      ++pos;
    }

    // Skip whitespace:
    skipSpace();

    return true;
  }

  void lex() {
    // Skip whitespace:
    skipSpace();

    // Skip comment(s):
    while (testAndSkipComment()) {
    }

    if (pos == input.size()) {
      eof = true;
      return;
    }

    TokenKind kind;
    size_t start, end;

    if (isdigit(input[pos])) {
      start = pos;
      end = pos + 1;

      while (end < input.size() && isdigit(input[end])) {
        ++end;
      }

      kind = EXPR_TK_CONSTANT;
    } else if (isalpha(input[pos])) {
      start = pos;
      end = pos + 1;

      while (end < input.size() && isalnum(input[end])) {
        ++end;
      }

      kind = EXPR_TK_VARIABLE;
    } else if (input[pos] == '$') {
      start = pos + 1;
      if (start == input.size() || !isalpha(input[start])) {
        llvm_unreachable("Invalid parameter name");
      }

      end = start + 1;
      while (end < input.size() && isalnum(input[end])) {
        ++end;
      }

      kind = EXPR_TK_PARAMETER;
    } else {
      // Single character tokens:
      switch (input[pos]) {
      case '(':
        kind = EXPR_TK_LPAREN;
        break;
      case ')':
        kind = EXPR_TK_RPAREN;
        break;
      case '*':
        kind = EXPR_TK_TIMES;
        break;
      case '-':
        kind = EXPR_TK_MINUS;
        break;
      case '+':
        kind = EXPR_TK_PLUS;
        break;
      default:
        llvm_unreachable("Invalid token");
      }

      start = pos;
      end = pos + 1;
    }

    current = {kind, input.substr(start, end - start), start, end};
    pos = end;
  }
};

struct ExprParser {
  explicit ExprParser(const std::string &source) : L(source) {}

  ExprRef parse() {
    if (atEOF())
      return ExprRef();

    // Make the lexer 'L' process the first token in the input:
    nextToken();

    ExprRef result = parseTerm();
    if (!atEOF()) {
      llvm_unreachable("Dangling input after expression");
    }

    return result;
  }

private:
  ExprLexer L;

  bool atEOF() const { return L.atEOF(); }

  Token currentToken() const { return L.getCurrent(); }

  TokenKind curKind() const { return L.getCurrent().kind; }

  void nextToken() { L.next(); }

  void consumeExpected(TokenKind k) { L.consumeExpected(k); }

  ExprRef parseTerm() {
    ExprRef result = parseProduct();

    while (curKind() == EXPR_TK_PLUS || curKind() == EXPR_TK_MINUS) {
      optype op = (curKind() == EXPR_TK_PLUS) ? PLUS : MINUS;

      nextToken();

      ExprRef expr = parseProduct();
      result = std::make_shared<BinOp>(op, result, expr);
    }

    return result;
  }

  ExprRef parseProduct() {
    ExprRef result = parseAtom();

    while (curKind() == EXPR_TK_TIMES) {
      nextToken();

      ExprRef expr = parseAtom();
      result = std::make_shared<BinOp>(TIMES, result, expr);
    }

    return result;
  }

  ExprRef parseAtom() {
    switch (curKind()) {
    case EXPR_TK_MINUS: {
      nextToken();
      ExprRef expr = parseAtom();
      return std::make_shared<Neg>(expr);
    }
    case EXPR_TK_VARIABLE: {
      std::string name = currentToken().lexeme;
      nextToken();
      return std::make_shared<Variable>(name);
    }
    case EXPR_TK_PARAMETER: {
      std::string name = currentToken().lexeme;
      nextToken();
      return std::make_shared<Parameter>(name);
    }
    case EXPR_TK_CONSTANT: {
      std::string val = currentToken().lexeme;
      nextToken();
      return std::make_shared<Constant>(stoul(val));
    }
    case EXPR_TK_LPAREN: {
      nextToken();
      ExprRef expr = parseTerm();
      consumeExpected(EXPR_TK_RPAREN);
      return expr;
    }
    default:
      llvm_unreachable("Unexpected token");
    }
  }
};

} // namespace ranges
} // namespace teckyl

#endif // TECKYL_TC_INFERENCE_EXPRESSION_PARSER_H_
