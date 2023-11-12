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
    # TODO: MOve those ops to the backend
    if inputs.device == "cpu":
        inputs_np = np.asarray(inputs.data, dtype=np.float32)
        targets_np = np.asarray(targets.data, dtype=np.int64)
    else:
        inputs_np = np.asarray(inputs.data.get(), dtype=np.float32)
        targets_np = np.asarray(targets.data.get(), dtype=np.int64)

    num_classes = inputs_np.shape[1]
    # TODO: Add arange to tensor
    indices = Tensor(np.arange(len(targets_np)) * num_classes + targets_np, requires_grad=False, device=inputs.device)

    inputs = inputs.reshape(-1)
    our_log_probs = inputs.take(indices)
    loss = our_log_probs.neg()

    if reduce == "mean":
        loss = loss.mean()

    if reduce == "sum":
        loss = loss.sum()

    return loss


def cross_entropy(inputs: Tensor, targets: Tensor):
    """Cross entropy loss.

    Args:
        input: List of Scalar values (raw logits).
        target: Indice of the target class.

    Returns:
        Scalar value of the loss.
    """
    # assert target < inputs.shape[1], "Invalid target"
    log_logits = inputs.log_softmax()
    return nllloss(log_logits, targets)
