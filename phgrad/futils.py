from typing import Optional

from phgrad.engine import Tensor
from phgrad import types


def one_hot(tensor: Tensor, num_classes: Optional[int] = None) -> Tensor:
    """One hot encoding."""
    assert (tensor.dtype == types.int32) or (
        tensor.dtype == types.int64
    ), f"Tensor must be of type int32 or int64, got {tensor.dtype}"

    if num_classes is None:
        num_classes = int(tensor.max().first_item) + 1

    if tensor.dims == 1:
        out_shape = [tensor.shape[0], num_classes]
    else:
        out_shape = list(tensor.shape) + [num_classes]

    # TODO: I think we can do better than this
    index_tensor = tensor.reshape((*tensor.shape, 1))

    out = Tensor.zeros(out_shape, device=tensor.device)
    out.scatter_add(index_tensor, 1, axis=-1)
    return out

