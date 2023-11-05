from typing import Optional

import numpy as np

from .engine import PensorTensor as pensor


def argmax(tensor: pensor, dim: Optional[int] = None) -> int:
    """Returns the index of the maximum value of a tensor."""
    return np.argmax(tensor.data, axis=dim)


