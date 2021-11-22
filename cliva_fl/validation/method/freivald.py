import torch, sys
from cliva_fl.utils import tensors_close
import torch.nn.functional as F

global global_diff

def freivald(A, B, C, bias=None, rtol=1e-05, atol=1e-08, n_check=1):
    """
    Freivalds' algorithm to check if AB = C

    Avoid errors because of precision in float32 with '> 1e-5'
    REF: https://discuss.pytorch.org/t/numerical-difference-in-matrix-multiplication-and-summation/28359
    """
    if bias is None: bias = 0.
    R = torch.round(torch.rand(B.shape[1], n_check))
    ABr = torch.matmul(A, torch.matmul(B, R))
    Cr = torch.matmul(torch.sub(C, bias), R)
    res = tensors_close(Cr, ABr, rtol=rtol, atol=atol)
    if not res:
        tola = (Cr - ABr).abs()
        tolr = ((Cr - ABr)/Cr).abs()
        print(A.shape, B.shape, C.shape, 'atol:', tola.max().item(), 'rtol:', tolr.max().item())
    return res
