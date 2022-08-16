#!/bin/bash

# buck build -c fbcode.platform=platform009 -c fbcode.static_nccl=1 @mode/opt //kineto/libkineto:kineto_stress_test

MPIPATH=/usr/local/fbcode/platform009/bin/mpirun
BINPATH=~/fbsource/fbcode/buck-out/gen/kineto/libkineto/kineto_stress_test
NUMRANKS=2

$MPIPATH -np $NUMRANKS $BINPATH $HOME/fbsource/fbcode/kineto/libkineto/stress_test/stress_test_uvm_nccl.json
