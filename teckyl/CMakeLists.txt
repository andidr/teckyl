add_custom_target(Teckyl)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(teckyl
  tc/lang/lexer.h
  tc/lang/tree.cpp
  tc/lang/tree.h
  tc/lang/lexer.cc
  tc/lang/parser.h
  tc/lang/tree_views.h
  tc/lang/builtins.h
  tc/lang/parser.cc
  tc/lang/error_report.h
  tc/lang/sema.h
  tc/lang/inference/expr.h
  tc/lang/inference/expr.cpp
  tc/lang/inference/ranges.h
  tc/lang/inference/ranges.cpp
  tc/utils/compiler_options.h
  Exception.h
  lang_affine.h
  lang_extras.h
  HeaderGen.h
  HeaderGen.cpp
  MLIRAffineExprGen.h
  MLIRGen.cpp
  MLIRGen.h
  main.cc
  patterns.h
  PrefixedOStream.h)

target_compile_options(teckyl PRIVATE -fexceptions -fno-rtti)

target_include_directories(teckyl PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  "${CMAKE_SOURCE_DIR}/llvm-project/mlir/include"
  "${CMAKE_SOURCE_DIR}/llvm-project/llvm/include"
  "${CMAKE_BINARY_DIR}/llvm-project/mlir/include"
  "${CMAKE_BINARY_DIR}/llvm-project/llvm/include"
  "${CMAKE_BINARY_DIR}/llvm-project/llvm/tools/mlir/include")

install(TARGETS teckyl RUNTIME DESTINATION bin)

target_link_libraries(teckyl
  PRIVATE
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRTransforms
    MLIRLinalgEDSC
    MLIREDSC
    MLIRLinalgOps)

