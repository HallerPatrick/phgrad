
## Benchmark

MNIST 10 Epochs on MacBook Pro
* SGD: ~12.7 seconds -> 94 Accuracy
* Adam: ~24 seconds -> 97 Accuracy

CPU: AMD Ryzen 7 5800X 8-Core Processor
GPU: GeForce RTX 3060 Ti
MNIST 10 Epochs

Commit: bdfa7448e75e69c3be4007a98e3c5053c9de189d
* SGD(CPU): 8.97 sec
* Adam(CPU): 15.65 sec
* SGD(GPU): 45.13 sec
* Adam(GPU): 51.08 sec

Commit: next, "Caching forward results"
Note: Looks like main compute time is not wasted on inefficient logsoftmax, but probably device moving and IO

* SGD(GPU): 44.31 sec, with SGD optimization: 44.07 sec, without backward: 9.95 sec!
    -> Most time spend in backward pass!
    -> Most time spend in backward pass of `Take` (26 secs!)
    -> Optimized, now 18 secs in total
* Adam(GPU): 50.77 sec



==== Summary ====

Tensors created
cuda: 56262
cpu: 187550
Total tensors: 243812

Func calls
<class 'phgrad.backends.cuda.Transpose'>: 37520
<class 'phgrad.backends.cuda.MatMul'>: 37520
<class 'phgrad.backends.cuda.ReLU'>: 18760
<class 'phgrad.backends.cuda.LogSoftmax'>: 18750
<class 'phgrad.backends.cuda.Reshape'>: 18750
<class 'phgrad.backends.cuda.Take'>: 18750
<class 'phgrad.backends.cuda.Neg'>: 18750
<class 'phgrad.backends.cuda.Mean'>: 18750
Total func calls: 187550
