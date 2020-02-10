# Teckyl: An MLIR frontend for Tensor Operations

Teckyl is released under the the Apache License v2.0 with LLVM
Exceptions. For details see the file LICENSE.

This project is based on llvm-project. See the llvm-project
subdirectory for details about the licsensing.

Teckyl also uses code from the Tensor Comprehensions repository at
https://github.com/facebookresearch/TensorComprehensions, licensed
exclusively under the Apache Licence, version 2.0 (i.e., without LLVM
exceptions). See the file teckyl/tc/LICENSE for details.

## Building Teckyl

After cloning the repository, make sure that all submodules are up to
date by executing

  ```git submodule init && git submodule update```

at the project root. You may then create a build directory, initialize
the build files by invoking `cmake` and running `make` to build Teckyl
as follows:

  * ``mkdir build``
  * ``cd build``
  * ``cmake ..``
  * ``make teckyl`` or ``make -j<N> teckyl``, where `<N>` is the
    number of cores of your build system

## Running Teckyl

The build instructions above produce the `teckyl` binary in `bin`. The
program takes at least two parameters: the name of an input file with
tensor expressions and the `-emit` option that specifies what Teckyl
should generate. Currently, Teckyl emits either a TC AST (`-emit=ast`)
or MLIR (`-emit=mlir`).

You can give the frontend a quick test by generating MLIR for one of
the examples in `tests/inputs`, e.g., by running

  ``./bin/teckyl -emit=mlir ../tests/inputs/mv_explicit.tc``

from the build directory.