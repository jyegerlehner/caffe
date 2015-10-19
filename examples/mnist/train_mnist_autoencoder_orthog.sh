#!/usr/bin/env sh

../../build/caffe_debug/tools/caffe train \
  --solver=examples/mnist/mnist_autoencoder_solver_orthog.prototxt
