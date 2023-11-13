import os
from collections import defaultdict

tensor_creations = defaultdict(int)
func_calls = defaultdict(int)
forward_time = defaultdict(int)
backward_time = defaultdict(int)

DEBUG = int(os.getenv("DEBUG", 0))

def print_summary():
    print("==== Summary ====")
    print("1. Tensors created")
    for k, v in tensor_creations.items():
        print(f"\t{k}: {v}")

    print(f"Total tensors: {sum(tensor_creations.values())}")

    print("2. Func calls")
    for k, v in func_calls.items():
        print(f"\t{k}: {v}")

    print(f"Total func calls: {sum(func_calls.values())}")

    print("3. Forward time")
    for k, v in forward_time.items():
        print(f"\t{k}: {v}")

    print(f"Total forward time: {sum(forward_time.values())}")

    print("4. Backward time")
    for k, v in backward_time.items():
        print(f"\t{k}: {v}")

    print(f"Total backward time: {sum(backward_time.values())}")

