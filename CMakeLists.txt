cmake_minimum_required(VERSION 3.4.3)

# For some reason, it is not sufficient to simply set
# LLVM_ENABLE_PROJECTS before add_subdirectory() without the CACHE
# option.
set(LLVM_ENABLE_PROJECTS "" CACHE STRING
  "Semicolon-separated list of projects to build, or \"all\". The mlir project is added by default.")
list(APPEND LLVM_ENABLE_PROJECTS "mlir")

# MLIR links some whole-archive libraries by name and they
# must be on the path. This may not be necessary in future versions of MLIR.
link_directories(${CMAKE_BINARY_DIR}/llvm-project/llvm/lib)

add_subdirectory(llvm-project/llvm)
add_subdirectory(teckyl)
add_subdirectory(teckyl/tc/lang/inference)
