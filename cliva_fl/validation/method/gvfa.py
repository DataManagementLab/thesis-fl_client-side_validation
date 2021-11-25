import torch, sys
from cliva_fl.utils import tensors_close
import torch.nn.functional as F

def gvfa(A, B, C, bias=None, rtol=1e-05, atol=1e-08, n_check= 1, float_type=torch.float32):
    """
    Gaussian Variant of Freivalds' Algorithm for Efficient and Reliable Matrix Product Verification to check if AB = C

    ArXiv: https://arxiv.org/abs/1705.10449
    """
    if bias is None: bias = 0.
    means = torch.zeros(B.shape[1], n_check, dtype=float_type)
    stds = torch.ones(B.shape[1], n_check, dtype=float_type)
    r = torch.normal(mean=means, std=stds)
    ABr = torch.matmul(A.to(float_type),torch.matmul(B.to(float_type), r))
    Cr = torch.matmul(torch.sub(C, bias).to(float_type), r)
    res = tensors_close(Cr, ABr, rtol=rtol, atol=atol)
    # if not res:
    #     tola = (Cr - ABr).abs()
    #     tolr = ((Cr - ABr)/Cr).abs()
    #     print(A.shape, B.shape, C.shape, 'atol:', tola.max().item(), 'rtol:', tolr.max().item())
    return res
    # return tensors_close(ABr, Cr, atol=atol)
