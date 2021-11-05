import torch, time
from flow.utils import tensors_close

def matmul(A, B, C, bias=None, rtol=1e-05, atol=1e-08):
    """
    Normal forward pass
    """
    if bias is None: bias = 0.
    C_ = torch.mm(A,B) + bias
    return tensors_close(C, C_, rtol=rtol, atol=atol)
