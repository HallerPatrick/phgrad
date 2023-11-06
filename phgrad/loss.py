from typing import List

import numpy as np

from .engine import Tensor

def nllloss(inputs: Tensor, targets: Tensor, reduce="mean"):
    """Negative log likelihood loss.

    Args:
        input: List of Scalar values.
        target: Indice of the target class.

    Returns:
        Scalar value of the loss.
    """
    assert reduce in ["mean", "sum"], "Invalid reduce"
    
    # TODO
    # If targets are not one-hot encoded, convert them to one-hot.
    # if len(np.squeeze(targets.data).shape) == 1:
    #     print("One hot encoding")
    #     targets = np.zeros((targets.shape[0], inputs.shape[1]))
    #     targets[np.arange(targets.shape[0]), targets] = 1

    loss = inputs.take(targets, dim=1).neg()

    if reduce == "mean":
        loss = loss.mean()

    if reduce == "sum":
        loss = loss.sum()

    return loss


def cross_entropy(inputs: Tensor, target: int):
    """Cross entropy loss.

    Args:
        input: List of Scalar values (raw logits).
        target: Indice of the target class.

    Returns:
        Scalar value of the loss.
    """
    # assert target < inputs.shape[1], "Invalid target"
    log_logits = inputs.log_softmax()
    return nllloss(log_logits, target)