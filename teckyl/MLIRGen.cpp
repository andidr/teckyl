#include "teckyl/MLIRGen.h"
#include "teckyl/MLIRAffineExprGen.h"
#include "teckyl/lang_affine.h"
#include "teckyl/lang_extras.h"
#include "teckyl/patterns.h"

#include "teckyl/tc/lang/sema.h"
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Linalg/EDSC/Builders.h>
#include <mlir/Dialect/Linalg/EDSC/Intrinsics.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/StandardTypes.h>

namespace teckyl {

static const char *getTypeAsString(mlir::Type t) {
  if (t.isF16())
    return "f16";
  else if (t.isF32())
    return "f32";
  else if (t.isF64())
    return "f64";
  else if (t.isInteger(8))
    return "i8";
  else if (t.isInteger(16))
    return "i16";
  else if (t.isInteger(32))
    return "i32";
  else if (t.isInteger(64))
    return "i64";
  else if (t.isIndex())
    return "index";
  llvm_unreachable("Cannot determine name for type");
}

static inline bool isMLIRFloatType(mlir::Type &t) {
  return t.isF16() || t.isF32() || t.isF64();
}

// Returns the total size in bits of the float type `t`. Throws an
// exception if `t` is not a float type.
static inline unsigned int getMLIRFloatTypeBits(mlir::Type &t) {
  if (t.isF16())
    return 16;
  if (t.isF32())
    return 32;
  if (t.isF64())
    return 64;
  llvm_unreachable("Not a float type");
}

// Returns the size in bits of the mantissa of the float type
// `t`. Throws an exception if `t` is not a float type.
static inline unsigned int getMLIRFloatTypeMantissaBits(mlir::Type &t) {
  if (t.isF16())
    return 10;
  if (t.isF32())
    return 23;
  if (t.isF64())
    return 52;

  llvm_unreachable("Not a float type");
}

// Returns the total size in bits of the integer type `t`. Throws an
// exception if `t` is not a integer type.
static inline unsigned int getMLIRIntTypeBits(mlir::Type &t) {
  if (t.isa<mlir::IntegerType>())
    return t.cast<mlir::IntegerType>().getWidth();

  llvm_unreachable("Not an integer type");
}

static inline bool isMLIRIntType(mlir::Type &t) {
  return t.isInteger(8) || t.isInteger(16) || t.isInteger(32) ||
         t.isInteger(64);
}

using IteratorBoundsMap =
    std::map<std::string, std::pair<mlir::Value, mlir::Value>>;

// Kinds of tensor expression iterators
enum IteratorKind {
  // Iterator appears on the left hand side (and may also appear at
  // the right hand side)
  LHS,

  // Iterator appears only on the right hand side
  RHSOnly
};

// Collects the set of iterators of a comprehensions by listing all
// identifiers and retaining only those that are not in the symbol
// table `symTab`.
static std::map<std::string, IteratorKind> collectIterators(
    const lang::Comprehension &comprehension,
    const llvm::ScopedHashTable<llvm::StringRef, mlir::Value> &symTab) {
  std::map<std::string, IteratorKind> iterators;

  for (const lang::Ident &lhsIndex : comprehension.indices())
    iterators.emplace(lhsIndex.name(), IteratorKind::LHS);

  mapRecursive(comprehension.rhs(), [&](const lang::TreeRef &t) {
    if (t->kind() == lang::TK_IDENT) {
      std::string name = lang::Ident(t).name();

      if (iterators.find(name) == iterators.end() && symTab.count(name) == 0) {
        iterators.emplace(name, IteratorKind::RHSOnly);
      }
    }
  });

  return iterators;
}

class MLIRGenBase {
public:
  MLIRGenBase(mlir::MLIRContext *context,
              const std::string &filename = "unknown file")
      : builder(context), filename(filename){};

  virtual ~MLIRGenBase() = default;

  mlir::OpBuilder &getBuilder() { return builder; }

protected:
  mlir::OpBuilder builder;
  std::string filename;

  // Translates a TC float type to an MLIR float type
  mlir::FloatType getFloatType(int kind) {
    switch (kind) {
    case lang::TK_DOUBLE:
      return builder.getF64Type();
    case lang::TK_FLOAT:
      return builder.getF32Type();
    case lang::TK_FLOAT16:
      return builder.getF16Type();
    case lang::TK_FLOAT32:
      return builder.getF32Type();
    case lang::TK_FLOAT64:
      return builder.getF64Type();
    default:
      llvm_unreachable("Not a float type");
    }
  }

  mlir::Type getScalarType(int kind) {
    switch (kind) {
    case lang::TK_DOUBLE:
    case lang::TK_FLOAT:
    case lang::TK_FLOAT16:
    case lang::TK_FLOAT32:
    case lang::TK_FLOAT64:
      return getFloatType(kind);
    case lang::TK_INT8:
      return builder.getIntegerType(8);
    case lang::TK_INT16:
      return builder.getIntegerType(16);
    case lang::TK_INT32:
      return builder.getIntegerType(32);
    case lang::TK_INT64:
      return builder.getIntegerType(64);
    case lang::TK_SIZET:
      return builder.getIndexType();
    default:
      llvm_unreachable("Unsupported type");
    }
  }

  // Returns the element type of `v` if `v` is a MemRef value,
  // otherwise the function returns the type of `v`.
  mlir::Type getElementType(const mlir::Value &v) {
    mlir::Type type = v.getType();

    if (type.isa<mlir::MemRefType>())
      return type.cast<mlir::MemRefType>().getElementType();
    else
      return type;
  }

  // Returns the rank of the type of `v`, if `v` is a MemRef
  // value. Otherwise an error occurs.
  int64_t getRank(const mlir::Value &v) {
    mlir::Type type = v.getType();

    if (type.isa<mlir::MemRefType>())
      return type.cast<mlir::MemRefType>().getRank();
    else
      llvm_unreachable("Can only determine rank for MemRef");
  }

  // Translates a TC tensor type into an MLIR tensor type. If the
  // original type is a scalar type, a scalar MLIR type is returned.
  mlir::Type getTensorType(const lang::TensorType &tensorType) {
    mlir::Type scalarType = getScalarType(tensorType.scalarType());
    size_t ndims = tensorType.dims().size();

    if (ndims > 0) {
      // Build a MemRef type with the correct number of dimensions,
      // but leave size of dimensions undefined return
      return mlir::MemRefType::get(std::vector<int64_t>(ndims, -1), scalarType);
    } else {
      return scalarType;
    }
  }

  // Translates a TC source location to an MLIR source location
  mlir::FileLineColLoc loc(const lang::SourceRange &r) {
    return builder
        .getFileLineColLoc(builder.getIdentifier(filename), r.startLine(),
                           r.endLine())
        .cast<mlir::FileLineColLoc>();
  }
};

// Convert the value `v` to type `t` if such a conversion is possible
// and lossless. Returns true if the conversion is successful,
// otherwise false.
static bool convertValue(mlir::OpBuilder &builder, mlir::Value &v, mlir::Type t,
                         mlir::Location location) {
  mlir::Type tV = v.getType();

  if (tV == t)
    return true;

  if (isMLIRFloatType(tV) && isMLIRFloatType(t) &&
      getMLIRFloatTypeBits(tV) < getMLIRFloatTypeBits(t)) {
    v = builder.create<mlir::FPExtOp>(location, v, t);
    return true;
  } else if (isMLIRIntType(tV) && isMLIRIntType(t) &&
             getMLIRIntTypeBits(tV) < getMLIRIntTypeBits(t)) {
    // TODO: When adding support for unsigned integers, use
    // ZeroExtendIOp
    v = builder.create<mlir::SignExtendIOp>(location, v, t);
    return true;
  } else if (isMLIRIntType(tV) && isMLIRFloatType(t)) {
    unsigned int intBits = getMLIRIntTypeBits(tV);
    unsigned int mantissaBits = getMLIRFloatTypeMantissaBits(t);

    if (intBits <= mantissaBits) {
      // FIXME: This is only correct for signed integers
      v = builder.create<mlir::SIToFPOp>(location, v, t);
      return true;
    }
  }

  return false;
}

// Align types of two values: If a and b are of different types, the
// function attempts to convert the type with less precision to the
// type with higher precision. Only lossless conversions are
// performed.
//
// Upon success, the function returns true (i.e., if the types were
// already aligned or if an alignment was successful). Otherwise, the
// function returns false.
static bool alignTypes(mlir::OpBuilder &builder, mlir::Value &a, mlir::Value &b,
                       mlir::Location location) {
  mlir::Type tA = a.getType();
  mlir::Type tB = b.getType();

  if (tA == tB)
    return true;

  if (isMLIRFloatType(tA) && isMLIRFloatType(tB)) {
    if (getMLIRFloatTypeBits(tA) < getMLIRFloatTypeBits(tB))
      return convertValue(builder, a, tB, location);
    else
      return convertValue(builder, b, tA, location);
  } else if (isMLIRIntType(tA) && isMLIRIntType(tB)) {
    if (getMLIRIntTypeBits(tA) < getMLIRIntTypeBits(tB))
      return convertValue(builder, a, tB, location);
    else
      return convertValue(builder, b, tA, location);
  } else if (isMLIRIntType(tA) && isMLIRFloatType(tB)) {
    unsigned int intBits = getMLIRIntTypeBits(tA);
    unsigned int mantissaBits = getMLIRFloatTypeMantissaBits(tB);

    if (intBits <= mantissaBits)
      return convertValue(builder, a, tB, location);
  } else if (isMLIRFloatType(tA) && isMLIRIntType(tB)) {
    unsigned int intBits = getMLIRIntTypeBits(tB);
    unsigned int mantissaBits = getMLIRFloatTypeMantissaBits(tA);

    if (intBits <= mantissaBits)
      return convertValue(builder, b, tA, location);
  }

  return false;
}

// Builds a binary operation from `lhs` and `rhs` associated to the
// specified location. If both values are float values, the newly
// created operation is `FOpTyp` and if both values are integer
// values, `IOpTy` is instantiated. If the values have different types
// or if they are neither floats nor integers, an error occurs.
template <typename FOpTy, typename IOpTy>
mlir::Value buildBinaryExprFromValues(mlir::OpBuilder &builder, mlir::Value lhs,
                                      mlir::Value rhs,
                                      mlir::FileLineColLoc location) {
  if (!alignTypes(builder, lhs, rhs, location)) {
    std::stringstream ss;

    ss << "Operands for binary expression have different types: "
       << getTypeAsString(lhs.getType()) << " and "
       << getTypeAsString(rhs.getType());

    mlirgen::SourceException err(location, ss.str());
    THROW_OR_ASSERT(err);
  }

  mlir::Type resType = lhs.getType();

  if (isMLIRFloatType(resType)) {
    return builder.create<FOpTy>(location, lhs, rhs);
  } else if (isMLIRIntType(resType)) {
    return builder.create<IOpTy>(location, lhs, rhs);
  } else {
    mlirgen::SourceException err(
        location, "Cannot create binary operation: Unsupported operand type");
    THROW_OR_ASSERT(err);
  }
}

// Builds MLIR expressions without control flow from tensor
// expressions
class MLIRValueExprGen : public MLIRGenBase {
public:
  MLIRValueExprGen(mlir::MLIRContext *context,
                   llvm::ScopedHashTable<llvm::StringRef, mlir::Value> &symTab,
                   const std::string &filename = "unknown filename")
      : MLIRGenBase(context, filename), symTab(symTab) {}

  MLIRValueExprGen(mlir::OpBuilder &_builder,
                   llvm::ScopedHashTable<llvm::StringRef, mlir::Value> &symTab,
                   const std::string &filename = "unknown filename")
      : MLIRGenBase(_builder.getContext(), filename), symTab(symTab) {
    this->builder.setInsertionPoint(_builder.getInsertionBlock(),
                                    _builder.getInsertionPoint());
  }

  // Builds a binary MLIR expression from a TC expression. Creates an
  // operation of type `FOpTy` if the operands are floats or an
  // operation of type `IOpTy` if the operands are integers. If the
  // operands have different types or if they are neither integers nor
  // floats, an error occurs.
  template <typename FOpTy, typename IOpTy>
  mlir::Value buildBinaryExpr(const lang::TreeRef &t) {
    return buildBinaryExprFromValues<FOpTy, IOpTy>(
        builder, buildExpr(t->trees().at(0)), buildExpr(t->trees().at(1)),
        loc(t->range()));
  }

  // Builds a constant from a string `s` with the same type as
  // `targetType`
  virtual mlir::Value buildConstant(const std::string &cst,
                                    const mlir::Type &targetType,
                                    const mlir::Location &location) {
    if (targetType.isa<mlir::FloatType>()) {
      mlir::FloatType floatType = targetType.cast<mlir::FloatType>();

      if (floatType.isF16()) {
        return builder.create<mlir::ConstantFloatOp>(
            location, llvm::APFloat(llvm::APFloat::IEEEhalf(), cst), floatType);
      } else if (floatType.isF32()) {
        return builder.create<mlir::ConstantFloatOp>(
            location, llvm::APFloat(llvm::APFloat::IEEEsingle(), cst),
            floatType);
      } else if (floatType.isF64()) {
        return builder.create<mlir::ConstantFloatOp>(
            location, llvm::APFloat(llvm::APFloat::IEEEdouble(), cst),
            floatType);
      } else {
        llvm_unreachable("Could not build constant: Unknown float type");
      }
    } else if (targetType.isa<mlir::IntegerType>()) {
      mlir::IntegerType iType = targetType.cast<mlir::IntegerType>();

      std::istringstream iss(cst);
      int64_t icst;

      if (!(iss >> icst)) {
        mlirgen::Exception err("Could not build integer constant");
        THROW_OR_ASSERT(err);
      }

      return builder.create<mlir::ConstantIntOp>(location, icst,
                                                 iType.getWidth());
    } else if (targetType.isa<mlir::IndexType>()) {
      std::istringstream iss(cst);
      int64_t icst;

      // FIXME: Check if constant fits into platform-dependent index
      // type
      if (!(iss >> icst)) {
        mlirgen::Exception err("Could not build index constant");
        THROW_OR_ASSERT(err);
      }

      return builder.create<mlir::ConstantIndexOp>(location, icst);
    } else {
      llvm_unreachable("Could not build constant: Unsupported target type");
    }
  }

  // Builds an MLIR constant from a TC constant. The type of the
  // constant is preserved.
  //
  // Throws an exception if the TC type cannot be expressed in MLIR.
  virtual mlir::Value buildConstant(const lang::Const &cst) {
    mlir::Type targetType = getScalarType(cst.type()->kind());
    return buildConstant(cst.value(), targetType, builder.getUnknownLoc());
  }

  // Builds a MLIR value corresponding to the TC identifier `i`.
  virtual mlir::Value buildIdent(const lang::Ident &i) {
    return symTab.lookup(i.name());
  }

  // Builds an MLIR load operation indexing the tensor that
  // corresponds to `ident` using the symbols corresponding to the
  // identifiers from `indices`.
  virtual mlir::LoadOp
  buildIndexLoadExpr(const lang::Ident &ident,
                     const lang::ListView<lang::Ident> &indices) {
    std::vector<mlir::Value> argVals;

    for (const lang::Ident &arg : indices) {
      auto subexpr = buildIdent(arg);
      argVals.push_back(subexpr);
    }

    mlir::Value tensor = symTab.lookup(ident.name());

    return builder.create<mlir::LoadOp>(loc(ident.range()), tensor, argVals);
  }

  // Builds an MLIR load operation indexing the tensor that
  // corresponds to `ident` using the expressions passed in `indices`.
  virtual mlir::LoadOp
  buildIndexLoadExpr(const lang::Ident &ident,
                     const lang::ListView<lang::TreeRef> &indices) {
    std::vector<mlir::Value> argVals;

    for (const lang::TreeRef &arg : indices) {
      auto subexpr = buildExpr(arg);
      argVals.push_back(subexpr);
    }

    mlir::Value tensor = symTab.lookup(ident.name());

    return builder.create<mlir::LoadOp>(loc(ident.range()), tensor, argVals);
  }

  // Translates a TC access expression into an MLIR load operation.
  virtual mlir::LoadOp buildIndexLoadExpr(const lang::Access &a) {
    return buildIndexLoadExpr(a.name(), a.arguments());
  }

  // Builds an MLIR store operation writing the value `valueToStore`
  // to the tensor corresponds to `ident` indexed using the symbols
  // corresponding to the identifiers from `indices`.
  virtual mlir::StoreOp
  buildIndexStoreExpr(mlir::Value &valueToStore, const lang::Ident &ident,
                      const lang::ListView<lang::Ident> &indices) {
    mlir::FileLineColLoc location(loc(ident.range()));
    mlir::Value tensor = symTab.lookup(ident.name());

    std::vector<mlir::Value> argVals;

    for (const lang::Ident &idx : indices) {
      auto subexpr = buildIdent(idx);
      argVals.push_back(subexpr);
    }

    mlir::StoreOp ret =
        builder.create<mlir::StoreOp>(location, valueToStore, tensor, argVals);

    mlir::Type elementType = ret.getMemRefType().getElementType();

    if (elementType != valueToStore.getType()) {
      std::stringstream ss;

      ss << "Assignment of a value of type "
         << getTypeAsString(valueToStore.getType())
         << " to a RHS value of type " << getTypeAsString(elementType);

      mlirgen::SourceException err(location, ss.str());
      THROW_OR_ASSERT(err);
    }

    return ret;
  }

  // Translates a TC expression into an MLIR expression
  virtual mlir::Value buildExpr(const lang::TreeRef &t) {
    switch (t->kind()) {
    case '+':
      return buildBinaryExpr<mlir::AddFOp, mlir::AddIOp>(t);
    case '-':
      return buildBinaryExpr<mlir::SubFOp, mlir::SubIOp>(t);
    case '*':
      return buildBinaryExpr<mlir::MulFOp, mlir::MulIOp>(t);
    case '/':
      return buildBinaryExpr<mlir::DivFOp, mlir::SignedDivIOp>(t);
    case lang::TK_NUMBER:
    case lang::TK_CONST:
      return buildConstant(lang::Const(t));
    case lang::TK_IDENT:
      return buildIdent(lang::Ident(t));
    case lang::TK_ACCESS:
      return buildIndexLoadExpr(lang::Access(t));
    default:
      std::stringstream ss;
      ss << "Unknown tree type: '" << (int)t->kind() << "'";
      std::cerr << ss.str() << std::endl;
      mlirgen::SourceException err(loc(t->range()), ss.str());
      THROW_OR_ASSERT(err);
    }
  }

  // Translates a map from identifiers to TC RangeContraints to a map
  // from identifiers to pairs of MLIR values for the respective
  // bounds
  virtual IteratorBoundsMap
  translateIteratorBounds(const IteratorRangeMap &langBounds) {
    IteratorBoundsMap mlirBounds;

    for (const std::pair<std::string, lang::RangeConstraint> &langBound :
         langBounds) {
      std::string iteratorName = langBound.first;
      const lang::RangeConstraint &constraint = langBound.second;

      mlir::Value lowBound = buildExpr(constraint.start());
      mlir::Value upBound = buildExpr(constraint.end());

      // Convert bounds to Index values if necessary.
      //
      // FIXME: Index has a platform-dependent width, which may be
      // lower than the width of the converted integer type and
      // silently truncate the value, leading to incorrect code.
      if (!lowBound.getType().isIndex()) {
        lowBound = builder.create<mlir::IndexCastOp>(
            loc(constraint.range()), builder.getIndexType(), lowBound);
      }

      if (!upBound.getType().isIndex()) {
        upBound = builder.create<mlir::IndexCastOp>(
            loc(constraint.range()), builder.getIndexType(), upBound);
      }

      mlirBounds.insert({iteratorName, {lowBound, upBound}});
    }

    return mlirBounds;
  }

protected:
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> &symTab;
};

// Builds MLIR expressions without control flow from tensor
// expressions. The difference with MLIRValueExprGen is that entire
// subtrees of the tensor expression can be mapped to MLIR values
// (e.g., to map sub-expressions to block or function arguments or to
// avoid re-generation of known sub-expressions).
class MLIRMappedValueExprGen : public MLIRValueExprGen {
public:
  MLIRMappedValueExprGen(
      mlir::OpBuilder &_builder,
      const std::map<lang::TreeId, mlir::Value> &valMap,
      llvm::ScopedHashTable<llvm::StringRef, mlir::Value> &symTab,
      const std::string &filename = "unknown filename")
      : MLIRValueExprGen(_builder, symTab, filename), valMap(valMap) {}

  virtual mlir::Value buildExpr(const lang::TreeRef &t) override {
    auto idxIt = valMap.find(t->id());
    if (idxIt != valMap.end())
      return idxIt->second;
    else
      return MLIRValueExprGen::buildExpr(t);
  }

protected:
  const std::map<lang::TreeId, mlir::Value> &valMap;
};

class MLIRGenImpl : protected MLIRGenBase {
public:
  MLIRGenImpl(mlir::MLIRContext *context, const MLIRGenOptions &options,
              const std::string &filename = "unknown file")
      : MLIRGenBase(context, filename), options(options) {}

  // Builds a FuncOp for a definition `def`
  mlir::FuncOp buildFunction(const std::string &name, const lang::Def &def) {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symTab);
    std::vector<mlir::Type> argTypes;

    // Add parameters for symbolic tensor dimensions
    std::set<std::string> sizeParams = collectDimSizeParams(def);

    // Add tensor parameters
    for (lang::Param param : def.params()) {
      lang::TensorType tensorType = param.tensorType();
      mlir::Type mlirTensorType = getTensorType(tensorType);
      argTypes.push_back(mlirTensorType);
    }

    // Add output parameters
    std::map<std::string, size_t> outputRanks = collectOutputRanks(def);

    for (lang::Param param : def.returns()) {
      lang::TensorType tcTensorType = param.tensorType();
      std::string name = param.ident().name();

      if (param.typeIsInferred()) {
        std::stringstream ss;

        ss << "Type for output tensor " << name << " not specified";

        mlirgen::SourceException err(loc(param.range()), ss.str());
        THROW_OR_ASSERT(err);
      }

      // Check that used dimensions correspond to the declared
      // dimensions
      if (outputRanks.find(name) != outputRanks.end()) {
        size_t declaredDims = tcTensorType.dims().size();

        if (declaredDims != outputRanks[name]) {
          std::stringstream ss;
          ss << "Output tensor " << name << " has been declared with "
             << declaredDims << " dimensions, "
             << "but is indexed with " << outputRanks[name] << " "
             << "dimensions";
          mlirgen::Exception err(ss.str());
          THROW_OR_ASSERT(err);
        }
      }

      mlir::Type mlirTensorType =
          mlir::MemRefType::get(std::vector<int64_t>(outputRanks[name], -1),
                                getScalarType(tcTensorType.scalarType()));

      argTypes.push_back(mlirTensorType);
    }

    mlir::FunctionType func_type =
        builder.getFunctionType(argTypes, llvm::None);

    mlir::FuncOp funcOp =
        mlir::FuncOp::create(loc(def.range()), name, func_type);

    mlir::FuncOp function(funcOp);
    mlir::Block &entryBlock = *function.addEntryBlock();

    builder.setInsertionPointToStart(&entryBlock);

    // Add all arguments to symbol table
    {
      size_t i = 0;

      // Add parameters for symbolic tensor dimensions to symbol table
      //
      // The sizes are not passed explicitly as function arguments,
      // but correspond to the dimensions of the input / output
      // tensors. For each size constant, choose the dimension of one
      // tensor as the defining representative.
      auto checkOrDefineSizeSymbol = [&](const lang::Param &param,
                                         mlir::BlockArgument &arg) {
        size_t dimIdx = 0;
        for (const lang::TreeRef &dim : param.tensorType().dims()) {
          if (dim->kind() == lang::TK_IDENT) {
            lang::Ident ident(dim);

            if (symTab.count(ident.name()) == 0) {
              // Use this as a repesentative for the size dimension
              mlir::Value sizeParamVal =
                  builder.create<mlir::DimOp>(loc(def.range()), arg, dimIdx);
              symTab.insert(ident.name(), sizeParamVal);
            }
          }

          dimIdx++;
        }
      };

      // Adds an entry for the tensor to `paramSpecs`, mapping tensor
      // names to their specification
      auto addParamSpec = [&](const lang::Param &param) {
        paramSpecs.insert({param.ident().name(), param.tensorType()});
      };

      // Process inputs
      for (lang::Param param : def.params()) {
        mlir::BlockArgument arg = funcOp.getArgument(i++);
        symTab.insert(param.ident().name(), arg);
        checkOrDefineSizeSymbol(param, arg);
        addParamSpec(param);
      }

      // Process outputs
      for (lang::Param param : def.returns()) {
        mlir::BlockArgument arg = funcOp.getArgument(i++);
        symTab.insert(param.ident().name(), arg);
        checkOrDefineSizeSymbol(param, arg);
        addParamSpec(param);
      }
    }

    for (const lang::Comprehension &comprehension : def.statements())
      buildComprehension(comprehension);

    builder.create<mlir::ReturnOp>(loc(def.range()));

    return function;
  }

private:
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symTab;
  std::map<const std::string, lang::TensorType> paramSpecs;
  const MLIRGenOptions options;

  // Used for tensor initialization
  enum NeutralElement { Zero = 0, One = 1 };

  // Builds a loop nest with one loop per iterator from `iterators`
  // using the bounds from `mlirIteratorBounds`.
  //
  // If `innermost` is non-NULL, a reference to the innermost loop is
  // stored in `*innermost`.
  mlir::scf::ForOp buildLoopNest(const std::vector<std::string> &iterators,
                                 const IteratorBoundsMap &mlirIteratorBounds,
                                 const mlir::Location &location,
                                 mlir::scf::ForOp *innermost = nullptr) {
    mlir::scf::ForOp outermost;
    mlir::Value step = builder.create<mlir::ConstantIndexOp>(location, 1);

    // Build loop nest for all involved iterators
    for (const auto &it : iterators) {
      mlir::scf::ForOp loop = builder.create<mlir::scf::ForOp>(
          location, mlirIteratorBounds.at(it).first,
          mlirIteratorBounds.at(it).second, step);

      if (!outermost)
        outermost = loop;

      // Create symbol table entry to map iterator names to induction
      // variables
      symTab.insert(it, loop.getInductionVar());

      if (innermost)
        *innermost = loop;

      builder.setInsertionPointToStart(loop.getBody());
    }

    return outermost;
  }

  // Builds a linalg.generic operation that initializes the specified
  // tensor with the specified value.
  void buildTensorInitialization(const std::string &tensorName,
                                 mlir::Value tensorVal,
                                 const lang::ListView<lang::Ident> &indexes,
                                 mlir::Location location, NeutralElement value,
                                 const IteratorRangeMap &langItBounds) {
    MLIRValueExprGen exprGen(builder, symTab, filename);
    mlir::Type elementType = getElementType(tensorVal);
    size_t rank = getRank(tensorVal);
    std::vector<mlir::IteratorType> iteratorTypes(rank,
                                                  mlir::IteratorType::Parallel);
    mlir::Value output;

    // Check if the bounds for the iterators used to index the output
    // tensor are equal to the size of the indexed dimensions.
    //
    // If this is the case, just use the output tensor as the output
    // memref for the linalg.generic operation.
    //
    // If the iterator ranges do not match the output tensor
    // dimensions, create a view with a one-to-one mapping from the
    // iteration domain to the tensor elements.
    if (comprehensionLHSIteratorDomainsMatchTensorDimensions(
            paramSpecs, langItBounds, tensorName, indexes)) {
      output = tensorVal;
    } else {
      std::vector<mlir::Value> offsets;
      std::vector<mlir::Value> sizes;
      std::vector<mlir::Value> strides{
          rank, exprGen.buildConstant("1", builder.getIndexType(), location)};

      for (const lang::Ident &index : indexes) {
        mlir::Value lb =
            exprGen.buildExpr(langItBounds.at(index.name()).start());
        mlir::Value ub = exprGen.buildExpr(langItBounds.at(index.name()).end());

        offsets.push_back(lb);

        // lb and ub are of type index; convert to integer, subtract and
        // convert back to index
        //
        // FIXME: Size of Index is platform-dependent, so this might be
        // a lossy conversion
        mlir::Value lbInt = builder.create<mlir::IndexCastOp>(
            location, builder.getIntegerType(64), lb);
        mlir::Value ubInt = builder.create<mlir::IndexCastOp>(
            location, builder.getIntegerType(64), ub);

        mlir::Value sizeInt =
            builder.create<mlir::SubIOp>(location, ubInt, lbInt);
        mlir::Value size = builder.create<mlir::IndexCastOp>(
            location, builder.getIndexType(), sizeInt);

        sizes.push_back(size);
      }

      output = builder.create<mlir::SubViewOp>(location, tensorVal, offsets,
                                               sizes, strides);
    }

    mlir::Value cstVal;

    switch (value) {
    case NeutralElement::Zero:
      cstVal = exprGen.buildConstant("0", elementType, location);
      break;
    case NeutralElement::One:
      cstVal = exprGen.buildConstant("1", elementType, location);
      break;
    }

    builder.create<mlir::linalg::FillOp>(location, output, cstVal);
  }

  // Collects all access expressions that are descendants of t in an
  // arbitrary order
  std::vector<lang::Access> collectTensorAccessesSeq(const lang::TreeRef &t) {
    std::vector<lang::Access> res;

    // Collect all tensor accesses in subexpressions
    mapRecursive(t, [&](const lang::TreeRef &e) {
      if (e->kind() == lang::TK_ACCESS)
        res.push_back(lang::Access(e));
    });

    return res;
  }

  // Builds the core of a comprehension (e.g., just the actual
  // compitation without the initialization broadcasting the neutral
  // element for default-initialized reductions). This is the fallback
  // routine for comprehensions with possibly non-affine accesses.
  void buildLoopReductionCore(const lang::Comprehension &c, mlir::Value tensor,
                              const std::vector<std::string> &iteratorsSeq,
                              const IteratorRangeMap &langItBounds,
                              mlir::Location location) {
    MLIRValueExprGen exprGen(builder, symTab, filename);

    IteratorBoundsMap mlirItBounds =
        exprGen.translateIteratorBounds(langItBounds);

    mlir::Block *currBlock = builder.getInsertionBlock();

    mlir::scf::ForOp innermost;
    buildLoopNest(iteratorsSeq, mlirItBounds, location, &innermost);

    exprGen.getBuilder().setInsertionPointToStart(innermost.getBody());

    // Build expression for RHS of assignment
    mlir::Value rhsVal = exprGen.buildExpr(c.rhs());
    mlir::Value accu;
    mlir::Value assignmentVal;

    switch (c.assignment()->kind()) {
    case lang::TK_PLUS_EQ:
    case lang::TK_PLUS_EQ_B:
      accu = exprGen.buildIndexLoadExpr(c.ident(), c.indices());
      assignmentVal = buildBinaryExprFromValues<mlir::AddFOp, mlir::AddIOp>(
          exprGen.getBuilder(), rhsVal, accu, loc(c.range()));
      break;
    case lang::TK_TIMES_EQ:
    case lang::TK_TIMES_EQ_B:
      accu = exprGen.buildIndexLoadExpr(c.ident(), c.indices());
      assignmentVal = buildBinaryExprFromValues<mlir::MulFOp, mlir::MulIOp>(
          exprGen.getBuilder(), rhsVal, accu, loc(c.range()));
      break;
    case '=':
      assignmentVal = rhsVal;
      break;
    default:
      llvm_unreachable("Unsupported operator");
    }

    mlir::Type elementType = getElementType(symTab.lookup(c.ident().name()));

    if (!convertValue(exprGen.getBuilder(), assignmentVal, elementType,
                      loc(c.range()))) {
      std::stringstream ss;

      ss << "Operand for assignment cannot be converted to element type of the "
            "target tensor: "
         << "cannot convert " << getTypeAsString(assignmentVal.getType())
         << " to " << getTypeAsString(elementType);

      mlirgen::SourceException err(loc(c.range()), ss.str());
      THROW_OR_ASSERT(err);
    }

    exprGen.buildIndexStoreExpr(assignmentVal, c.ident(), c.indices());

    // Restore insertion point to point after the outermost loop
    builder.setInsertionPointToEnd(currBlock);
  }

  // Creates an instance of OP_T from c if CHECK_FUNC returns
  // true. The order of the input operands to OP_T is the canonical
  // order provided by CHECK_FUNC and the order of output operands is
  // the same as in outputs.
  template <typename OP_T, int num_inputs,
            bool (*CHECK_FUNC)(const lang::Comprehension &c,
                               size_t (*canonical_order)[num_inputs])>
  bool tryBuildSpecializedLinalgOp(
      const lang::Comprehension &c,
      llvm::ArrayRef<mlir::edsc::StructuredIndexed> inputs,
      llvm::ArrayRef<mlir::edsc::StructuredIndexed> outputs) {

    llvm::SmallVector<mlir::Value, 4> rearranged;
    size_t canon[num_inputs];

    // Check if c is of correct type and determine canonical order for
    // input operands
    if (CHECK_FUNC(c, &canon)) {
      for (int i = 0; i < num_inputs; i++)
        rearranged.push_back(inputs[canon[i]]);

      for (mlir::edsc::StructuredIndexed o : outputs)
        rearranged.push_back(o);

      mlir::ValueRange operands(
          mlir::ArrayRef<mlir::Value>{rearranged.begin(), rearranged.end()});

      builder.create<OP_T>(loc(c.range()), mlir::TypeRange{}, operands);

      return true;
    }

    return false;
  }

  // Tries to build a linalg structured operations from c and the
  // provided inputs / outputs.
  bool tryBuildSpecializedLinalgOp(
      const lang::Comprehension &c,
      llvm::ArrayRef<mlir::edsc::StructuredIndexed> inputs,
      llvm::ArrayRef<mlir::edsc::StructuredIndexed> outputs) {
    return tryBuildSpecializedLinalgOp<mlir::linalg::MatmulOp, 2,
                                       pattern::isMatmulComprehension>(
               c, inputs, outputs) ||
           tryBuildSpecializedLinalgOp<mlir::linalg::MatmulOp, 2,
                                       pattern::isDefinitMatmulComprehension>(
               c, inputs, outputs) ||
           tryBuildSpecializedLinalgOp<mlir::linalg::MatvecOp, 2,
                                       pattern::isMatvecComprehension>(
               c, inputs, outputs) ||
           tryBuildSpecializedLinalgOp<mlir::linalg::MatvecOp, 2,
                                       pattern::isDefinitMatvecComprehension>(
               c, inputs, outputs);
  }

  // Builds the core of a comprehension (e.g., just the actual
  // compitation without the initialization broadcasting the neutral
  // element for default-initialized reductions) with affine
  // accesses. The check for affine accesses must be performed prior
  // to the call.
  void
  buildLinalgReductionCore(const lang::Comprehension &c, mlir::Value tensor,
                           const std::map<std::string, IteratorKind> &iterators,
                           const std::vector<std::string> &iteratorsSeq,
                           mlir::Location location) {
    std::vector<lang::Access> tensorAccesses =
        collectTensorAccessesSeq(c.rhs());
    std::vector<mlir::edsc::StructuredIndexed> inputs;
    std::vector<mlir::Value> inputTensorValues;
    std::set<std::string> accessedTensors;
    std::map<lang::TreeId, unsigned int> argIndexes;

    // Extract names of all tensors that are indexed on the rhs
    for (const lang::Access &access : tensorAccesses)
      accessedTensors.insert(access.name().name());

    // Add output tensor
    accessedTensors.insert(c.ident().name());

    // Create a mapping between iterators and their dimension index
    // for the affine expression for fast lookup
    std::map<std::string, unsigned int> iteratorDims;

    {
      unsigned int dim = 0;
      for (const std::string &it : iteratorsSeq)
        iteratorDims.emplace(it, dim++);
    }

    MLIRAffineExprGen affGen(builder.getContext(), iteratorDims);

    // Create one AffineExpr per access dimension of each tensor
    // access; keep a mapping between access expressions and the index
    // within the lists of input block arguments for the generated
    // linalg operation
    for (const lang::Access &a : tensorAccesses) {
      std::vector<mlir::AffineExpr> aff = affGen.buildAffineExpressions(a);

      mlir::Value tensorValue = symTab.lookup(a.name().name());
      inputTensorValues.push_back(tensorValue);
      mlir::edsc::StructuredIndexed tensorBase(tensorValue);
      mlir::edsc::StructuredIndexed tensorIndexed = tensorBase(aff);

      argIndexes.insert({a.id(), inputs.size()});
      inputs.push_back(tensorIndexed);
    }

    // Create a StructuredIndexed for the output tensor indexed by the
    // non-reduction dimensions
    std::vector<mlir::edsc::StructuredIndexed> outputs;
    {
      std::vector<mlir::AffineExpr> aff =
          affGen.buildAffineExpressions(c.indices());
      mlir::edsc::StructuredIndexed tensorHandle(tensor);
      mlir::edsc::StructuredIndexed tensorIndexed(tensorHandle(aff));
      outputs.push_back(tensorIndexed);
    }

    // Build iteration dimensions
    std::vector<mlir::IteratorType> iteratorTypes;

    for (const std::string &it : iteratorsSeq) {
      if (iterators.at(it) == IteratorKind::LHS)
        iteratorTypes.push_back(mlir::IteratorType::Parallel);
      else
        iteratorTypes.push_back(mlir::IteratorType::Reduction);
    }

    mlir::edsc::ScopedContext sc(builder, location);

    // Region builder for the body of the linalg.generic
    // operation. The block arguments are the tensor elements from the
    // access expressions and the value at the current position in the
    // output tensor.
    //
    // Generate MLIR expressions for the rhs tensor expression of the
    // comprehension, but use mappings to block arguments for all
    // access expressions.
    auto regionBuilder = [&](mlir::ValueRange blockArgs) {
      // Prepare mapping from lang::Tree IDs to block Arguments representing the
      // tensor reads
      std::map<lang::TreeId, mlir::Value> valMap;

      for (auto it : argIndexes)
        valMap.insert({it.first, blockArgs[it.second]});

      MLIRMappedValueExprGen gen(mlir::edsc::ScopedContext::getBuilderRef(),
                                 valMap, symTab, filename);
      mlir::Value rhsVal = gen.buildExpr(c.rhs());

      // Accumulator for output tensor is always the last argument
      mlir::Value accu = blockArgs[blockArgs.size() - 1];
      mlir::Value res;

      // Build the operator for the reduction and store final value
      // for the reduction step in res
      switch (c.assignment()->kind()) {
      case lang::TK_PLUS_EQ:
      case lang::TK_PLUS_EQ_B:
        res = buildBinaryExprFromValues<mlir::AddFOp, mlir::AddIOp>(
            gen.getBuilder(), rhsVal, accu, loc(c.range()));
        break;
      case lang::TK_TIMES_EQ:
      case lang::TK_TIMES_EQ_B:
        res = buildBinaryExprFromValues<mlir::MulFOp, mlir::MulIOp>(
            gen.getBuilder(), rhsVal, accu, loc(c.range()));
        break;
      case '=':
        res = rhsVal;
        break;
      default:
        llvm_unreachable("Unsupported operator");
      }

      mlir::Type elementType = getElementType(tensor);

      if (!convertValue(gen.getBuilder(), res, elementType, loc(c.range()))) {
        std::stringstream ss;

        ss << "Operand for assignment cannot be converted to element type of "
              "the target tensor: "
           << "cannot convert " << getTypeAsString(res.getType()) << " to "
           << getTypeAsString(elementType);

        mlirgen::SourceException err(loc(c.range()), ss.str());
        THROW_OR_ASSERT(err);
      }

      mlir::edsc::intrinsics::linalg_yield{res};
    };

    bool buildGeneric = true;

    if (options.specialize_linalg_ops) {
      if (tryBuildSpecializedLinalgOp(c, inputs, outputs))
        buildGeneric = false;
    }

    if (buildGeneric) {
      mlir::edsc::makeGenericLinalgOp(iteratorTypes, inputs, outputs,
                                      regionBuilder);
    }
  }

  // Builds the MLIR representation of a single comprehension
  void buildComprehension(const lang::Comprehension &c) {
    mlir::Location startLoc = loc(c.range());

    // New scope for iterators
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> var_scope(symTab);

    std::map<std::string, IteratorKind> iterators = collectIterators(c, symTab);
    std::set<std::string> iteratorSet;
    std::set<std::string> iteratorSetReduction;
    IteratorRangeMap langItBounds = collectExplicitIteratorBounds(c);

    for (const std::pair<std::string, IteratorKind> &it : iterators) {
      iteratorSet.insert(it.first);

      if (it.second == IteratorKind::RHSOnly)
        iteratorSetReduction.insert(it.first);
    }

    // Decide on an (arbitrary) order for the iterators for the loop
    // nest
    std::vector<std::string> iteratorsSeq;

    for (std::pair<std::string, IteratorKind> it : iterators)
      iteratorsSeq.push_back(it.first);

    const std::string &outTensorName = c.ident().name();
    mlir::Value outTensorVal = symTab.lookup(outTensorName);

    // Initialize output tensor for default-initialized reductions
    if (c.assignment()->kind() == lang::TK_PLUS_EQ_B) {
      buildTensorInitialization(outTensorName, outTensorVal, c.indices(),
                                startLoc, NeutralElement::Zero, langItBounds);
    } else if (c.assignment()->kind() == lang::TK_TIMES_EQ_B) {
      buildTensorInitialization(outTensorName, outTensorVal, c.indices(),
                                startLoc, NeutralElement::One, langItBounds);
    } else if (c.assignment()->kind() == lang::TK_MAX_EQ_B ||
               c.assignment()->kind() == lang::TK_MIN_EQ_B) {
      // TODO: Support max and min
      llvm_unreachable("Unsupported reduction");
    }

    // Build code for the actual computation
    //
    // Check if the reduction of the comprehension is eligible for a
    // linalg.generic operation. The requirements are:
    //
    // 1. All tensor indexing must be affine.
    //
    // 2. The existence of a direct mapping between iteration
    //    dimensions and tensor accesses. This requires that each
    //    iterator of the comprehension is referenced at least once
    //    for direct indexing. For example, this is the case for:
    //
    //      C(i, j) = A(i) + A(i / 2) + B(k)
    //
    //    since i, j, and k are all used for direct indexing at least
    //    once, while:
    //
    //      C(i, j) = A(i) + A(i / 2) + B(k+5)
    //
    //    would not meet the condition above, since k is never
    //    directly used index a tensor dimension.
    //
    // 3. Since the iteration domains are directly derived from the
    //    tensors dimensions, the bounds for the comprehension for
    //    iterators with direct indexing must match the size of the
    //    respective tensor dimension.
    //
    // Conditions 2 and 3 might be relaxed in the future in cases,
    // where it is possible to create subviews which restore the
    // conditions.
    if (options.body_op == MLIRGenOptions::BodyOp::ScfFor ||
        hasNonAffineIndexing(c.rhs(), iteratorSet) ||
        !allIteratorsIndexTensorDimension(iteratorSetReduction, c.rhs()) ||
        !directIteratorDomainsMatchTensorDimensions(c, paramSpecs)) {
      buildLoopReductionCore(c, outTensorVal, iteratorsSeq, langItBounds,
                             startLoc);
    } else {
      buildLinalgReductionCore(c, outTensorVal, iterators, iteratorsSeq,
                               startLoc);
    }
  }

  // Returns a map with one entry per output tensor specifying their
  // ranks for the TC definition `def`. If the same tensor is indexed
  // with multiple ranks (e.g., C(i, j) = ... and C(i, j, k) = ..., a fatal
  // error occurs.
  std::map<std::string, size_t> collectOutputRanks(const lang::Def &def) {
    std::set<std::string> outParamNames;
    std::map<std::string, size_t> ranks;

    for (const lang::Param &outParam : def.returns())
      outParamNames.insert(outParam.ident().name());

    for (const lang::Comprehension &compr : def.statements()) {
      std::string name = compr.ident().name();
      size_t rank = compr.indices().size();

      if (outParamNames.find(name) != outParamNames.end()) {
        auto it = ranks.find(name);

        if (it != ranks.end()) {
          if (it->second != rank) {
            mlirgen::Exception err("Multiple ranks found for output tensor " +
                                   name);
            THROW_OR_ASSERT(err);
          }
        } else {
          ranks.insert({name, rank});
        }
      }
    }

    return ranks;
  }
};

// Builds an MLIR function with the name `name` from the TC definition
// `def`.
mlir::FuncOp buildMLIRFunction(mlir::MLIRContext &context,
                               const std::string &name, const lang::Def &tc,
                               const MLIRGenOptions &options) {
  MLIRGenImpl generator(&context, options);
  return generator.buildFunction(name, tc);
}
} // namespace teckyl
