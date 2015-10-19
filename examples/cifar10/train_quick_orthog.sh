#!/usr/bin/env sh

#TOOLS=./build/tools
TOOLS=../../build/caffe_debug/tools

$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver_orthog.prototxt

# reduce learning rate by factor of 10 after 8 epochs
$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver_lr1_orthog.prototxt \
  --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate.h5
