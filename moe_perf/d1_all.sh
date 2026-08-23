#!/bin/bash
DIR=$(cd "$(dirname "$0")" && pwd)
bash "$DIR/d1_sweep.sh"
bash "$DIR/d1_nsys.sh" 1
bash "$DIR/d1_nsys.sh" 32
echo "D1_ALL_DONE"
