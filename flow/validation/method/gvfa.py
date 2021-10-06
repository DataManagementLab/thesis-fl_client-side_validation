import torch, sys
from flow.utils import tensors_close
import torch.nn.functional as F

def gvfa(A, B, C, bias=None, rtol=1e-05, atol=1e-08):
    """
    Gaussian Variant of Freivalds' Algorithm for Efficient and Reliable Matrix Product Verification to check if AB = C

    ArXiv: https://arxiv.org/abs/1705.10449
    """
    if bias is None: bias = 0.
    means = torch.zeros(B.shape[1])
    # stds = torch.arange(1, B.shape[1]+1)
    stds = torch.ones(B.shape[1])
    r = torch.normal(mean=means, std=stds)
    ABr = torch.mv(A,torch.mv(B, r))
    Cr = torch.mv(C - bias, r)
    # print(r.shape)
    # print(r)
    # print(ABr.shape)
    # print(Cr.shape)
    # print(bias.shape)
    # diff = ABr - Cr
    # print(diff)
    # sys.exit(0)
    rdiff = torch.max(ABr - Cr).item()
    global glob_rdiff
    if not 'glob_rdiff' in globals() or rdiff > glob_rdiff:
        glob_rdiff = rdiff
        print(glob_rdiff)
    # return torch.allclose(ABr, Cr, rtol=rtol, atol=atol)
    return tensors_close(ABr, Cr, atol=atol)
