#!/bin/bash
DIR=$(cd "$(dirname "$0")" && pwd)
bash "$DIR/d1_nsys.sh" 1
bash "$DIR/d1_nsys.sh" 32
echo "NSYS_REDO_DONE"
bash "$DIR/d2_tune.sh"
echo "CHAIN_DONE"
