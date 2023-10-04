#!/bin/bash
g++ \
  -g3 \
  -O0 \
  kineto_playground.cpp \
  -o main \
  -I/usr/local/cuda/include \
  -I../third_party/fmt/include \
  -I/usr/local/include/kineto \
  -L/usr/local/lib \
  -L/usr/local/cuda/lib64 \
  -lpthread \
  -lcuda \
  -lcudart \
  /usr/local/lib/libkineto.a \
  kplay_cu.o
