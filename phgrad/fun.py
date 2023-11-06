from typing import Optional

import numpy as np

from .engine import Tensor as Tensor


def argmax(tensor: Tensor, dim: Optional[int] = None) -> int:
    """Returns the index of the maximum value of a tensor."""
    return np.argmax(tensor.data, axis=dim)


