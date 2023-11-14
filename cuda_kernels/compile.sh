#!/bin/sh

# We dont usually have to compile the cuda kernels, as cupy is taking care of that.
# But if you want to compile them or quick debugging.
nvcc -c -o <kernel>.o <kernel>.cu -x cu -ccbin=/opt/cuda/bin
