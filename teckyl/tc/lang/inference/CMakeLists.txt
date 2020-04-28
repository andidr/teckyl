
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(transform
  expr.h
  expr.cpp
  ../lexer.cc # Required due to a dependency on 'lang::kindToString'.
  analysis.h
  transformation.h
  transformation.cpp
  expression_parser.h
  ranges.h    # Header file not required.
  ranges.cpp  # Source file not required.
  transform.cc)

target_compile_options(transform PRIVATE -fno-rtti)

target_include_directories(transform PRIVATE
   ${CMAKE_CURRENT_SOURCE_DIR}/../../../..
   "${CMAKE_SOURCE_DIR}/llvm-project/llvm/include"
   "${CMAKE_BINARY_DIR}/llvm-project/llvm/include")

install(TARGETS transform RUNTIME DESTINATION bin)
