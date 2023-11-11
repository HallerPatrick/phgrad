#!/bin/sh

python -m cProfile -o mnist.prof examples/mnist.py
python -m flameprof mnist.prof > mnist.svg
open -a "google chrome" mnist.svg
