#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

buck build -c fbcode.platform=platform010 -c fbcode.static_nccl=1 @mode/opt //kineto/libkineto:kineto_stress_test

MPIPATH=/usr/local/fbcode/platform010/bin/mpirun
BINPATH=~/fbsource/fbcode/buck-out/gen/kineto/libkineto/kineto_stress_test
NUMRANKS=2

$MPIPATH -np $NUMRANKS $BINPATH $HOME/fbsource/fbcode/kineto/libkineto/stress_test/stress_test_uvm_nccl.json
