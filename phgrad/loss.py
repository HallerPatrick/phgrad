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

    # Flatten the inputs and create an array of indices corresponding to the target classes
    # # TODO: MOve those ops to the backend
    if inputs.device == "cpu":
        inputs_np = np.asarray(inputs.data, dtype=np.float32)
        targets_np = np.asarray(targets.data, dtype=np.int64)
    else:
        inputs_np = np.asarray(inputs.data.get(), dtype=np.float32)
        targets_np = np.asarray(targets.data.get(), dtype=np.int64)

    
    # TODO: We should not use numpy here
    # Impl: arange
    indices = Tensor.arange(inputs.shape[0], device=inputs.device) * inputs.shape[1] + targets
    # indices = np.arange(inputs.shape[0]) * inputs.shape[1] + targets_np
    inputs = inputs.reshape(-1)
    our_log_probs = inputs.take(indices)
    loss = our_log_probs.neg()

    if reduce == "mean":
        loss = loss.mean()

    if reduce == "sum":
        loss = loss.sum()

    return loss


def cross_entropy(inputs: Tensor, targets: Tensor, dim: int = -1):
    """Cross entropy loss.

    Args:
        input: List of Scalar values (raw logits).
        target: Indice of the target class.

    Returns:
        Scalar value of the loss.
    """
    # assert target < inputs.shape[1], "Invalid target"
    log_logits = inputs.log_softmax(dim=dim)
    return nllloss(log_logits, targets)
