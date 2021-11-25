import torch, numpy, sys, time
from cliva_fl.utils import tensors_close

def submul(A, B, C, bias=None, rtol=1e-05, atol=1e-08, ratio=.5, shuffle=True):
    """
    Normal forward pass
    """
    if bias is None: bias = torch.zeros(B.shape[1])

    if shuffle:
        permut = torch.randperm(A.shape[0])[0:int(A.shape[0]*ratio)]
        Ap = A[permut]
        Cp = C[permut]
        tclose = torch.allclose(torch.matmul(Ap, B) + bias, Cp)
    else:
        frac_A = int(A.shape[0] * ratio)
        strt_A = torch.randint(0, A.shape[0] - frac_A,(1,)).item()
        A_ = torch.narrow(A, 0, strt_A, frac_A)
        C_ = torch.narrow(C, 0, strt_A, frac_A)
        tclose = tensors_close(torch.mm(A_, B) + bias, C_, rtol=rtol, atol=atol)
    
    return tclose
