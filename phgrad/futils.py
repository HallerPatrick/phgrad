from phgrad.engine import Tensor
from phgrad import types



def one_hot(tensor: Tensor, num_classes: int) -> Tensor:
    """One hot encoding."""
    assert (tensor.dtype == types.int32) or (tensor.dtype == types.int64), f"Tensor must be of type int32 or int64, got {tensor.dtype}"

    dims = list(tensor.shape)
    dims.append(num_classes)
    out = Tensor.zeros(dims, device=tensor.device)
    
    print(out.shape)
    print(tensor.shape)
    out.scatter_add(tensor, 1, axis=1)
    return out


