from typing import List

from .engine import Scalar

def nllloss(inputs: List[Scalar], target: int, reduce="mean"):
    """Negative log likelihood loss.

    Args:
        input: List of Scalar values.
        target: Indice of the target class.

    Returns:
        Scalar value of the loss.
    """
    assert reduce in ["mean", "sum"]
    assert target < len(inputs)

    loss = -inputs[target].value
    if reduce == "mean":
        loss /= len(inputs)

    return Scalar(loss)