add_custom_target(Teckyl)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LINK_COMPONENTS Support)

add_llvm_executable(teckyl main.cc)

install(TARGETS teckyl RUNTIME DESTINATION bin)

target_link_libraries(teckyl
  PRIVATE
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRTransforms)
