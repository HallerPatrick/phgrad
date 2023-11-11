from argparse import Namespace
from .cpu import ops_map

def backend_from_device(device: str):

    if device == "cpu":
        return Namespace(**ops_map)
