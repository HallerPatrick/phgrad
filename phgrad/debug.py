import os
from collections import defaultdict

tensor_creations = defaultdict(int)
func_calls = defaultdict(int)
forward_time = defaultdict(int)
backward_time = defaultdict(int)

DEBUG = int(os.getenv("DEBUG", 0))

def compressed_progress_bar(execution_time, total_time):
    """
    Generates a compressed progress bar with different characters representing different ranges.

    :param execution_time: Execution time of the function.
    :param total_time: Total execution time of all functions.
    :return: String representing the progress bar.
    """
    percentage = (execution_time / total_time) * 100
    symbols = "▏▎▍▌▋▊▉█"
    bar_max = 20  # Total number of characters in the bar

    full_chars = int(percentage // (100 / bar_max))
    remainder = percentage % (100 / bar_max)
    partial_char_index = int(remainder / (100 / bar_max * len(symbols)))

    bar = symbols[-1] * full_chars
    if full_chars < bar_max:
        bar += symbols[partial_char_index] + symbols[0] * (bar_max - full_chars - 1)

    return '[' + bar + '] ' + f'{percentage:.1f}%'


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
    total_time = sum(forward_time.values())
    # Sort by largest time
    forward_time_ = {k: v for k, v in sorted(forward_time.items(), key=lambda item: item[1], reverse=True)}
    for k, v in forward_time_.items():
        k = k.replace("class 'phgrad.backends.", "").replace("'", "")
        print(f"\t{k:>20}: {v:.3f} ({compressed_progress_bar(v, total_time)})")

    print(f"Total forward time: {sum(forward_time.values())}")

    total_time = sum(backward_time.values())
    backward_time_ = {k: v for k, v in sorted(backward_time.items(), key=lambda item: item[1], reverse=True)}
    print("4. Backward time")
    for k, v in backward_time_.items():
        print(f"\t{k:>20}: {v:.3f} ({compressed_progress_bar(v, total_time)})")

    print(f"Total backward time: {sum(backward_time.values())}")

