import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarker import benchmark

from phgrad import Tensor
from phgrad.utils import has_cuda_support
from phgrad.backends.cuda import _load_cuda_kernels

from cupyx.profiler import benchmark as cbenchmark

assert has_cuda_support(), "CUDA is not supported on this system"

dims = (1024, 1024, 1024)

num_times = 10000

@benchmark('Softmax(cpu)', num=num_times)
def cpu():
    t1 = Tensor.ones(dims)
    t2 = t1.softmax(dim=1)

# @benchmark('Softmax(cuda)', num=num_times)
# def cuda():
#     t1 = Tensor.ones(dims, device='cuda')
#     t1.softmax(dim=1)

@benchmark('Softmax(cuda_kernel)', num=num_times)
def cuda_kernel():
    t1 = Tensor.ones(dims, device='cuda')
    t1.softmax(dim=1)

if __name__ == "__main__":
    # cpu()
    # cuda()
    cuda_kernel()

