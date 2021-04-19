
def baseline(A, B, C, bias=None, rtol=1e-05, atol=1e-08, details=True):
    """
    Normal forward pass
    """
    if bias is None: bias = 0.
    C_ = torch.mm(A,B) + bias
    return torch.allclose(C, C_, rtol=rtol, atol=atol)
