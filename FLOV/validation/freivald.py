
def freivald(A, B, C, bias=None, rtol=1e-05, atol=1e-08, radom=True):
    """
    Freivalds' algorithm to check if AB = C

    Avoid errors because of precision in float32 with '> 1e-5'
    REF: https://discuss.pytorch.org/t/numerical-difference-in-matrix-multiplication-and-summation/28359
    """
    if bias is None: bias = 0.
    r_max = 509090009
    if random: r = (torch.randint(r_max, (B.shape[1],)) % 2).float()
    else: r = torch.ones(B.shape[1])
    ABr = torch.mv(A,torch.mv(B, r))
    Cr = torch.mv(C - bias, r)
    return torch.allclose(ABr, Cr, rtol=rtol, atol=atol)
