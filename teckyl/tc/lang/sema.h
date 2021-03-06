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
#ifndef TECKYL_TC_SEMA_H_
#define TECKYL_TC_SEMA_H_

#include <unordered_set>

#include "teckyl/PrefixedOStream.h"
#include "teckyl/tc/lang/builtins.h"
#include "teckyl/tc/lang/error_report.h"
#include "teckyl/tc/lang/inference/ranges.h"
#include "teckyl/tc/lang/tree.h"
#include "teckyl/tc/lang/tree_views.h"
#include "teckyl/tc/utils/compiler_options.h"

namespace lang {

// modified from Halide. It would be weird for Sema to take a halide
// dependency for this trivial functionality, and it allows us to
// modify the behavior in the future
struct TypeInfo {
  enum Code { Int, UInt, Float };
  TypeInfo(Code code_, uint8_t bits_) : code_(code_), bits_(bits_) {}
  TypeInfo(TreeRef scalar_type) {
    switch (scalar_type->kind()) {
#define TYPE_INFO_OPTION(tok, c, b)                                            \
  case tok:                                                                    \
    code_ = c;                                                                 \
    bits_ = b;                                                                 \
    break;
      TYPE_INFO_OPTION(TK_BOOL, UInt, 1)
      TYPE_INFO_OPTION(TK_UINT2, UInt, 2)
      TYPE_INFO_OPTION(TK_UINT4, UInt, 4)
      TYPE_INFO_OPTION(TK_UINT8, UInt, 8)
      TYPE_INFO_OPTION(TK_UINT16, UInt, 16)
      TYPE_INFO_OPTION(TK_UINT32, UInt, 32)
      TYPE_INFO_OPTION(TK_UINT64, UInt, 64)
      TYPE_INFO_OPTION(TK_INT2, Int, 2)
      TYPE_INFO_OPTION(TK_INT4, Int, 4)
      TYPE_INFO_OPTION(TK_INT8, Int, 8)
      TYPE_INFO_OPTION(TK_INT16, Int, 16)
      TYPE_INFO_OPTION(TK_INT32, Int, 32)
      TYPE_INFO_OPTION(TK_INT64, Int, 64)
      TYPE_INFO_OPTION(TK_FLOAT16, Float, 16)
      TYPE_INFO_OPTION(TK_FLOAT32, Float, 32)
      TYPE_INFO_OPTION(TK_FLOAT64, Float, 64)
      TYPE_INFO_OPTION(TK_FLOAT, Float, 32)
      TYPE_INFO_OPTION(TK_DOUBLE, Float, 64)
      TYPE_INFO_OPTION(TK_SIZET, UInt, 64)

#undef TYPE_INFO_OPTION
    default:
      ErrorReport err(scalar_type);
      err << "Unhandled TC scalar type: " << scalar_type;
      llvm_unreachable(err.what());
    }
  }
  int toScalarToken() const {
    switch (code()) {
    case UInt:
      switch (bits()) {
      case 1:
        return TK_BOOL;
      case 2:
        return TK_UINT2;
      case 4:
        return TK_UINT4;
      case 8:
        return TK_UINT8;
      case 16:
        return TK_UINT16;
      case 32:
        return TK_UINT32;
      case 64:
        return TK_UINT64;
      }
    case Int:
      switch (bits()) {
      case 2:
        return TK_INT2;
      case 4:
        return TK_INT4;
      case 8:
        return TK_INT8;
      case 16:
        return TK_INT16;
      case 32:
        return TK_INT32;
      case 64:
        return TK_INT64;
      }
    case Float:
      switch (bits()) {
      case 16:
        return TK_FLOAT16;
      case 32:
        return TK_FLOAT;
      case 64:
        return TK_DOUBLE;
      }
    }

    llvm_unreachable("Unknown type info?");
  }
  Code code() const { return code_; }
  uint8_t bits() const { return bits_; }
  bool is_float() const { return code_ == Float; }
  bool is_uint() const { return code_ == UInt; }

private:
  Code code_;
  uint8_t bits_;
};

static inline bool operator==(TypeInfo a, TypeInfo b) {
  return a.bits() == b.bits() && a.code() == b.code();
}

static inline TreeRef match_types(TreeRef a, TreeRef b) {
  TypeInfo ta(a);
  TypeInfo tb(b);
  if (ta == tb)
    return a;

  if (!ta.is_float() && tb.is_float()) {
    // int(a) * float(b) -> float(b)
    // uint(a) * float(b) -> float(b)
    return b;
  } else if (ta.is_float() && !tb.is_float()) {
    return a;
  } else if (ta.is_float() && tb.is_float()) {
    // float(a) * float(b) -> float(max(a, b))
    if (ta.bits() > tb.bits())
      return a;
    else
      return b;
  } else if (ta.is_uint() && tb.is_uint()) {
    // uint(a) * uint(b) -> uint(max(a, b))
    if (ta.bits() > tb.bits())
      return a;
    else
      return b;
  } else if (!ta.is_float() && !tb.is_float()) {
    // int(a) * (u)int(b) -> int(max(a, b))
    int bits = std::max(ta.bits(), tb.bits());
    return Compound::create(TypeInfo(TypeInfo::Int, bits).toScalarToken(),
                            a->range(), {});
  } else {
    ErrorReport err(b);
    err << "Could not match types: " << kindToString(ta.toScalarToken()) << ", "
        << kindToString(tb.toScalarToken());
    llvm_unreachable(err.what());
  }
}

/// Semantic analysis transforms the raw AST into a
/// typed and semantically correct tree.
/// Currently it:
/// - replaces TK_APPLY with TK_ACCESS nodes for tensor reads
/// - replace TK_APPLY with TK_BUILT_IN for built in functions
/// - checks that all variables are defined, and creates index/reduction
/// variable objects.
/// - checks that input variables are readonly.
struct Sema {
  explicit Sema(
      const tc::CompilerOptions &compilerOptions = tc::CompilerOptions())
      : compilerOptions(compilerOptions) {}

  TreeRef typeOfExpr(TreeRef ref) {
    if (expr_to_type.count(ref) == 0) {
      ErrorReport err(ref);
      err << "INTERNAL ERROR: type not in map for expression " << ref;
      llvm_unreachable(err.what());
    }
    return expr_to_type.at(ref);
  }

  // associate a type with this expression
  TreeRef withType(TreeRef expr, TreeRef type) {
    auto inserted = expr_to_type.emplace(expr, type).second;
    TC_ASSERT(expr, inserted);
    return expr;
  }

  TensorType expectTensorType(TreeRef loc, TreeRef typ) {
    if (typ->kind() != TK_TENSOR_TYPE) {
      ErrorReport err(loc);
      err << "expected a tensor but found a scalar";
      llvm_unreachable(err.what());
    }
    return TensorType(typ);
  }

  TreeRef matchAllTypes(TreeRef list, TreeRef matched_type = nullptr) {
    for (auto e : list->trees()) {
      if (!matched_type)
        matched_type = typeOfExpr(e);
      else
        matched_type = match_types(matched_type, typeOfExpr(e));
    }
    return matched_type;
  }

  TreeRef expectIntegral(TreeRef e) {
    if (TypeInfo(typeOfExpr(e)).code() == TypeInfo::Float) {
      ErrorReport err(e);
      err << " expected integral type but found "
          << kindToString(typeOfExpr(e)->kind());
      llvm_unreachable(err.what());
    }
    return e;
  }

  void expectBool(TreeRef anchor, int token) {
    if (token != TK_BOOL) {
      ErrorReport err(anchor);
      llvm_unreachable(err.what());
      err << "expected boolean but found " << kindToString(token);
    }
  }

  TreeRef expectBool(TreeRef exp) {
    expectBool(exp, typeOfExpr(exp)->kind());
    return exp;
  }

  TreeRef lookupVarOrCreateIndex(Ident ident) {
    TreeRef type = lookup(ident, false);
    if (!type) {
      // variable exp is not defined, so a reduction variable is created
      // a reduction variable index i
      type = indexType(ident);
      insert(index_env, ident, type, true);
      reduction_variables.push_back(ident);
    }
    return type;
  }

  TreeRef checkExp(TreeRef exp, bool allow_access) {
    switch (exp->kind()) {
    case TK_APPLY: {
      auto a = Apply(exp);
      if (!allow_access
          /* && live_input_names.count(a.name().name()) == 0 */) {
        // We want to allow access to inputs in this context, but it
        // isn't yet supported
        ErrorReport err(exp);
        err << "tensor accesses cannot be used in this context";
        llvm_unreachable(err.what());
      }

      // also handle built-in functions log, exp, etc.
      auto ident = a.name();
      if (builtin_functions.count(ident.name()) > 0) {
        auto nargs = builtin_functions[ident.name()];
        if (nargs != a.arguments().size()) {
          ErrorReport err(exp);
          err << "expected " << nargs << " but found " << a.arguments().size();
          llvm_unreachable(err.what());
        }
        auto args = checkExp(a.arguments(), allow_access);
        // [BUILTIN TYPE MATCHING]
        // for now we assume, dangerously, that all built in are just
        // float or double
        // numeric functions and should propagate their types like +, -, *,
        // div
        auto type = matchAllTypes(args, floatType(exp));
        return withType(BuiltIn::create(exp->range(), ident.name(), args, type),
                        type);
      }
      auto type = expectTensorType(ident, lookup(ident, true));
      if (type.dims().size() != a.arguments().size()) {
        ErrorReport err(a);
        err << "expected " << type.dims().size() << " dimensions but found "
            << a.arguments().size() << " dimensions.";
        llvm_unreachable(err.what());
      }
      auto checked = checkExp(a.arguments(), allow_access);
      for (auto t : checked->trees()) {
        expectIntegral(t);
      }

      {
        auto tt = type;
        auto output_annotation = annotated_output_types.find(ident.name());
        if (output_annotation != annotated_output_types.end())
          tt = TensorType(output_annotation->second);

        for (int i = 0; i < type.dims().size(); i++) {
          ranges_to_infer.addConstraints(
              std::make_shared<teckyl::ranges::Constant>(0),
              teckyl::ranges::Expr::fromTreeRef(a.arguments()[i],
                                                rangeParameters),
              teckyl::ranges::Expr::fromTreeRef(tt.dims()[i], rangeParameters));
        }
      }
      return withType(Access::create(exp->range(), ident, checked),
                      type.scalarTypeTree());
    } break;
    case TK_IDENT: {
      auto ident = Ident(exp);
      auto type = lookupVarOrCreateIndex(ident);
      if (type->kind() == TK_TENSOR_TYPE) {
        auto tt = TensorType(type);
        if (tt.dims().size() != 0) {
          ErrorReport err(exp);
          err << "expected a scalar but found a tensor expression.";
          llvm_unreachable(err.what());
        }
        return checkExp(Apply::create(ident.range(), ident,
                                      List::create(ident.range(), {})),
                        allow_access);
      }
      return withType(exp, type);
    } break;
    case '.': {
      auto s = Select(exp);
      auto ident = s.name();
      expectTensorType(ident, lookup(ident, true));
      return withType(exp, dimType(s));
    } break;
    case '+':
    case '-':
    case '*':
    case '/':
    case '%':
    case TK_MIN:
    case TK_MAX: {
      auto nexp =
          exp->map([&](TreeRef c) { return checkExp(c, allow_access); });
      return withType(nexp, matchAllTypes(nexp));
    } break;
    case TK_EQ:
    case TK_NE:
    case TK_GE:
    case TK_LE:
    case '<':
    case '>': {
      auto nexp =
          exp->map([&](TreeRef c) { return checkExp(c, allow_access); });
      // make sure the types match but the return type
      // is always bool
      matchAllTypes(nexp);
      return withType(nexp, boolType(exp));
    } break;
    case TK_AND:
    case TK_OR:
    case '!': {
      auto nexp =
          exp->map([&](TreeRef c) { return checkExp(c, allow_access); });
      expectBool(exp, matchAllTypes(nexp)->kind());
      return withType(nexp, boolType(exp));
    } break;
    case '?': {
      auto nexp =
          exp->map([&](TreeRef c) { return checkExp(c, allow_access); });
      expectBool(nexp->tree(0));
      auto rtype =
          match_types(typeOfExpr(nexp->tree(1)), typeOfExpr(nexp->tree(2)));
      return withType(nexp, rtype);
    }
    case TK_CONST: {
      auto c = Const(exp);
      return withType(exp, c.type());
    } break;
    case TK_CAST: {
      auto c = Cast(exp);
      auto nexp = checkExp(c.value(), allow_access);
      // currently this does not error, but we may want it to in the future
      match_types(typeOfExpr(nexp), c.type());
      return withType(Cast::create(c.range(), nexp, c.type()), c.type());
    }
    case TK_LIST: {
      return exp->map([&](TreeRef c) { return checkExp(c, allow_access); });
    } break;
    default:
      ErrorReport err(exp);
      err << "NYI - semantic checking for " << exp;
      llvm_unreachable(err.what());
    }
  }

  void addRangeParameters(TreeRef tensorType) {
    std::function<TreeRef(TreeRef)> addRecursive = [&](TreeRef t) -> TreeRef {
      if (t->kind() == TK_IDENT) {
        const std::string name = Ident(t).name();
        if (rangeParameters.count(name) == 0)
          rangeParameters.insert(name);
      }

      for (auto tt : t->trees())
        tt->map(addRecursive);

      return t;
    };

    addRecursive(tensorType);
  }

  // This is the entry function for semantic analysis. It is called by
  // tc2halide to associate type with each node of the tree and to also make
  // sure that the tree is sematically correct. For example: a variable
  // may not be input two times. Parser only verifies for the syntax but does
  // not check the semantics.
  //
  // It converts the TK_APPLY nodes to TK_ACCESS or TK_BUILT_IN
  //
  // The reduction variables are deduced and the objects are created for them
  // and they are appended to the tree
  //
  // Type checking is also done by small amount of code
  //
  // The method 'withType' is used to associate the type with a given node
  //
  TreeRef checkFunction(TreeRef func_) {
    auto func = Def(func_);
    auto params_ =
        checkList(func.params(), [&](TreeRef r) { return checkParam(r); });

    for (auto r : func.returns()) {
      if (!r.typeIsInferred()) {
        annotated_output_types.emplace(r.ident().name(), r.tensorType());
        addRangeParameters(r.tensorType());
	checkParam(r);
      }
    }

    // Everything has to be input or output. Keep track of the variables that
    // are either input/output. We will check that the statements have variables
    // from this list.
    for (auto p : func.params()) {
      nonTemporaries.insert(p.ident().name());
      inputParameters.insert(p.ident().name());
      if (!p.typeIsInferred()) {
        addRangeParameters(p.tensorType());
      }
    }
    for (auto r : func.returns()) {
      nonTemporaries.insert(r.ident().name());
      lookup(env, r. ident(), false);
    }

    auto statements_ =
        checkList(func.statements(), [&](TreeRef r) { return checkStmt(r); });
    auto returns_ =
        checkList(func.returns(), [&](TreeRef r) { return checkReturn(r); });
    auto r =
        Def::create(func.range(), func.name(), params_, returns_, statements_);

    rangeParameters.clear();

    return r;
  }

  TreeRef indexType(TreeRef anchor) {
    return createCompound(TK_INT32, anchor->range(), {});
  }

  TreeRef dimType(TreeRef anchor) { return indexType(anchor); }

  TreeRef floatType(TreeRef anchor) {
    return createCompound(TK_FLOAT, anchor->range(), {});
  }

  TreeRef boolType(TreeRef anchor) {
    return createCompound(TK_BOOL, anchor->range(), {});
  }

  void checkDim(Ident dim) { insert(env, dim, dimType(dim), false); }

  TreeRef checkTensorType(TreeRef type) {
    auto tt = TensorType(type);
    for (const auto &d : tt.dims()) {
      // dims may also be constants
      if (d->kind() == TK_IDENT)
        checkDim(Ident(d));
    }
    return type;
  }

  TreeRef checkParam(TreeRef param) {
    auto p = Param(param);
    TreeRef type_ = checkTensorType(p.type());
    insert(env, p.ident(), type_, true);
    live_input_names.insert(p.ident().name());
    return param;
  }

  TreeRef checkReturn(TreeRef ret) {
    auto r = Param(ret);
    TreeRef real_type = lookup(env, r.ident(), true);
    return ret;
  }

  TreeRef checkList(TreeRef list, std::function<TreeRef(TreeRef)> fn) {
    TC_ASSERT(list, list->kind() == TK_LIST);
    TreeList r;
    for (auto e : list->trees()) {
      r.push_back(fn(e));
    }
    return List::create(list->range(), std::move(r));
  }

  TreeRef checkRangeConstraint(RangeConstraint rc) {
    // RCs are checked _before_ the rhs of the TC, so
    // it is possible the index is not in the environment yet
    // calling lookupOrCreate ensures it exists
    lookupVarOrCreateIndex(rc.ident());
    // calling looking directly in the index_env ensures that
    // we are actually constraining an index and not some other variable
    lookup(index_env, rc.ident(), true);
    auto s = expectIntegral(checkExp(rc.start(), false));
    auto e = expectIntegral(checkExp(rc.end(), false));

    ranges_to_infer.addConstraints(
        teckyl::ranges::Expr::fromTreeRef(s, rangeParameters),
        teckyl::ranges::Expr::fromTreeRef(rc.ident(), rangeParameters),
        teckyl::ranges::Expr::fromTreeRef(e, rangeParameters));

    return RangeConstraint::create(rc.range(), rc.ident(), s, e);
  }

  TreeRef checkLet(Let l) {
    auto rhs = checkExp(l.rhs(), true);
    insert(let_env, l.name(), typeOfExpr(rhs), true);
    return Let::create(l.range(), l.name(), rhs);
  }

  TreeRef checkWhereClause(TreeRef ref) {
    if (ref->kind() == TK_LET) {
      return checkLet(Let(ref));
    } else if (ref->kind() == TK_EXISTS) {
      auto exp = checkExp(Exists(ref).exp(), true);
      return Exists::create(ref->range(), exp);
    } else {
      return checkRangeConstraint(RangeConstraint(ref));
    }
  }

  // Semantic checking for the statements/comprehensions in a TC Def.
  TreeRef checkStmt(TreeRef stmt_) {
    auto stmt = Comprehension(stmt_);

    const std::string name = stmt.ident().name();

    const auto tr = lookup(annotated_output_types, stmt.ident(), true);
    const auto tt = TensorType(tr);
    const auto dims = List(tt.dims());

    // register index variables (non-reductions)
    for (int i = 0; i < stmt.indices().size(); i++) {
      // for (const auto &index : stmt.indices()) {
      const auto &index = Ident(stmt.indices()[i]);
      ranges_to_infer.addRange(
          index.name(), std::make_shared<teckyl::ranges::Constant>(0),
          teckyl::ranges::Expr::fromTreeRef(dims[i], rangeParameters));
      auto typ = indexType(index);
      insert(index_env, index, typ, true);
    }

    // check that the input is not used for output - inputs are immutable
    if (inputParameters.count(name) > 0) {
      ErrorReport err(stmt_);
      err << "TC inputs are immutable";
      llvm_unreachable(err.what());
    }

    // make dimension variables for each dimension of the output tensor
    TreeList output_indices;
    int n = stmt.indices().size();
    for (int i = 0; i < n; ++i) {
      auto new_var =
          Ident::create(stmt.range(), name + "." + std::to_string(i));
      output_indices.push_back(new_var);
    }

    // where clauses are checked _before_ the rhs because they
    // introduce let bindings that are in scope for the rhs
    auto where_clauses_ = stmt.whereClauses().map(
        [&](TreeRef rc) { return checkWhereClause(rc); });

    TreeRef rhs_ = checkExp(stmt.rhs(), true);
    TreeRef scalar_type = typeOfExpr(rhs_);

    // if this statement will be returned and it is annotated in the return list
    // with a type (e.g. float(A,B)) then force the tensor to be that type
    // and check that the number of dimensions are consistent
    auto output_annotation = annotated_output_types.find(stmt.ident().name());
    if (output_annotation != annotated_output_types.end()) {
      auto tt = TensorType(output_annotation->second);
      auto matched_type = match_types(scalar_type, tt.scalarTypeTree());
      if (tt.scalarTypeTree()->kind() != matched_type->kind()) {
        ErrorReport err(stmt);
        err << " attempting to assign type "
            << kindToString(scalar_type->kind()) << " to narrower type "
            << kindToString(tt.scalarTypeTree()->kind())
            << " without an explicit cast";
        llvm_unreachable(err.what());
      }
      if (tt.dims().size() != stmt.indices().size()) {
        ErrorReport err(stmt);
        err << " tensor defined with " << stmt.indices().size()
            << " dimensions but declared as an output with " << tt.dims().size()
            << " dimensions.";
        llvm_unreachable(err.what());
      }
    }

    // After checking rhs and before creating lhs, we check if it is a reduction
    // without initialization (i.e., reduction operator without "!" suffix, and
    // lhs not defined previously).
    if (isUninitializedReductionOperation(stmt.assignment()) &&
        nullptr == lookup(stmt.ident(), false)) {
      ErrorReport err(stmt);
      std::string tk = kindToToken(stmt.assignment()->kind());
      err << "Reduction without initialization. If " << stmt.ident().name()
          << " is not pre-initialized before calling the TC function,"
          << " consider using the !-suffixed reduction operator " << tk
          << "! instead of " << tk;
      warn(err, compilerOptions);
    }

    auto type = TensorType::create(
        stmt.range(), scalar_type,
        List::create(stmt.range(), std::move(output_indices)));
    insert(env, stmt.ident(), type, false);

    // if we redefined an input, it is no longer valid for range expressions
    live_input_names.erase(stmt.ident().name());

    auto equivalent_statement_ = stmt.equivalent().map([&](Equivalent eq) {
      auto indices_ = eq.accesses().map(
          [&](TreeRef index) { return checkExp(index, true); });
      return Equivalent::create(eq.range(), eq.name(), indices_);
    });

    TreeRef assignment = stmt.assignment();
    // For semantic consistency we allow overwriting reductions like +=!
    // to be used in the language when there are no actual reduction dimensions.
    // Later compile stages assume that there is at least one reduction
    // dimension so if a reduction is specified and there are no reduction
    // dimensions, we revert back to assignment here.
    if (reduction_variables.size() == 0 && isNotInplace(assignment)) {
      assignment = Compound::create('=', assignment->range(), {});
    }

    if (reduction_variables.size() > 0 && stmt.assignment()->kind() == '=') {
      ErrorReport err(stmt);
      err << "this statement includes reduction variable '"
          << Ident(reduction_variables.back()).name()
          << "' but does not specify a reduction.";
      llvm_unreachable(err.what());
    }
    TreeRef reduction_variable_list =
        List::create(stmt.ident().range(), std::move(reduction_variables));
    TreeRef result = Comprehension::create(
        stmt.range(), stmt.ident(), stmt.indices(), stmt.assignment(), rhs_,
        where_clauses_, equivalent_statement_, reduction_variable_list);

    if (nonTemporaries.count(stmt.ident().name()) == 0) {
      ErrorReport err(stmt);
      err << stmt.ident().name() << " is not listed as an input or output to "
          << "this function. Temporaries tensors are not yet implemented";
      llvm_unreachable(err.what());
    }

    if (compilerOptions.printRanges) {
      std::stringstream ss;
      ss << stmt.range().filename() << ":" << stmt.range().startLine() << ": ";

      teckyl::PrefixedOStream pos(ss.str(), std::cout);
      pos << ranges_to_infer;
    }

    // clear the per-statement environments to get ready for the next statement
    index_env.clear();
    let_env.clear();

    ranges_to_infer.clear();

    return result;
  }

  static bool isUninitializedReductionOperation(TreeRef assignment) {
    switch (assignment->kind()) {
    case TK_PLUS_EQ:
    case TK_TIMES_EQ:
    case TK_MIN_EQ:
    case TK_MAX_EQ:
      return true;
    default:
      return false;
    }
  }

  bool isNotInplace(TreeRef assignment) {
    switch (assignment->kind()) {
    case TK_PLUS_EQ_B:
    case TK_TIMES_EQ_B:
    case TK_MIN_EQ_B:
    case TK_MAX_EQ_B:
      return true;
    default:
      return false;
    }
  }

  std::string dumpEnv() {
    std::stringstream ss;
    std::vector<std::pair<std::string, TreeRef>> elems(env.begin(), env.end());
    std::sort(elems.begin(), elems.end(),
              [](const std::pair<std::string, TreeRef> &t,
                 const std::pair<std::string, TreeRef> &t2) {
                return t.first < t2.first;
              });
    for (auto p : elems) {
      ss << p.first << ": " << p.second;
    }
    return ss.str();
  }

private:
  using Env = std::unordered_map<std::string, TreeRef>;

  void insert(Env &the_env, Ident ident, TreeRef value,
              bool must_be_undefined) {
    std::string name = ident.name();
    if (builtin_functions.count(name) > 0) {
      ErrorReport err(ident);
      err << "'" << name << "' is a built-in function and cannot be redefined";
      llvm_unreachable(err.what());
    }
    auto it = the_env.emplace(name, value);
    if (must_be_undefined && !it.second) {
      ErrorReport err(ident);
      err << name << " already defined";
      llvm_unreachable(err.what());
    }
  }

  TreeRef lookup(Ident ident, bool required) {
    TreeRef v = lookup(index_env, ident, false);
    if (!v)
      v = lookup(let_env, ident, false);
    if (!v)
      v = lookup(env, ident, required);
    return v;
  }

  TreeRef lookup(Env &the_env, Ident ident, bool required) {
    std::string name = ident.name();
    auto it = the_env.find(name);
    if (required && it == the_env.end()) {
      ErrorReport err(ident);
      err << "undefined variable " << name << " used here.";
      llvm_unreachable(err.what());
    }
    return it == the_env.end() ? nullptr : it->second;
  }

  TreeRef createCompound(int kind, const SourceRange &range, TreeList &&trees) {
    return Compound::create(kind, range, std::move(trees));
  }

  std::vector<TreeRef> reduction_variables; // per-statement
  Env index_env;                            // per-statement
  Env let_env; // per-statement, used for where i = <exp>

  Env env;                    // name -> type
  Env annotated_output_types; // name -> type, for all annotated returns types
  // identifiers that currently refer to an input tensor
  // values in these tensors are allowed in range expressions
  // if you write to an input, using it in a range expression is no longer
  // allowed
  std::unordered_set<std::string> live_input_names;

  std::unordered_map<TreeRef, TreeRef> expr_to_type;

  std::unordered_set<std::string> inputParameters;
  std::unordered_set<std::string> nonTemporaries;

  teckyl::ranges::InferenceProblem ranges_to_infer; // per-statement
  std::unordered_set<std::string> rangeParameters;  // per-function

  // TC compilation flow options.
  tc::CompilerOptions compilerOptions;
};
} // namespace lang

#endif // TECKYL_TC_SEMA_H_
