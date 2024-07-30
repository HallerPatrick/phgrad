import time

# Function decorator to benchmark the function
def benchmark(benchmark_name: str, num: int = 1):
    def inner(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            for _ in range(num):
                func(*args, **kwargs)
            end = time.time()
            print(f"{benchmark_name} executed {num} times in {end - start} seconds. Average time: {(end - start) / num} seconds per iteration")
        return wrapper
    return inner
    
