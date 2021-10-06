import torch, numpy, sys, time
from flow.utils import tensors_close

def submul(A, B, C, bias=None, rtol=1e-05, atol=1e-08, frac=0.5):
    """
    Normal forward pass
    """
    t1 = time.time()
    if bias is None: bias = torch.zeros(B.shape[1])
    
    frac_A = int(A.shape[0] * frac)
    frac_B = int(B.shape[1] * frac)

    if True:
        strt_A = torch.randint(0, A.shape[0] - frac_A,(1,)).item()
        strt_B = torch.randint(0, B.shape[1] - frac_B,(1,)).item()
        
        A__ = A[strt_A:strt_A+frac_A]
        B__ = B[:,strt_B:strt_B+frac_B]
        bias__ = bias[strt_B:strt_B+frac_B]
        C__ = C[strt_A:strt_A+frac_A][:,strt_B:strt_B+frac_B]
    else:
        choice_A = numpy.random.choice(
            torch.arange(0,A.shape[0]), 
            size=frac_A)
        choice_B = numpy.random.choice(
            torch.arange(0,B.shape[1]), 
            size=frac_B)
        
        A__ = A[choice_A]
        B__ = B[:,choice_B]
        bias__ = bias[choice_B]
        C__ = C[choice_A][:,choice_B]

    t2 = time.time()
    C_ = torch.mm(A__, B__) + bias__
    t3 = time.time()
    tclose = tensors_close(C__, C_, rtol=rtol, atol=atol)
    t4 = time.time()
    # C_ = torch.mm(A, B) + bias
    # t5 = time.time()
    
    if False and type(bias) is torch.nn.parameter.Parameter:
        print('t2-t1', t2-t1)
        print('t3-t2', t3-t2)
        print('t4-t3', t4-t3)
        print('t5-t4', t5-t4)
        # print('choice_A', len(choice_A))
        # print('choice_B', len(choice_B))
        print('A', A.shape)
        print('B', B.shape)
        print('bias', bias.shape)
        # print('C', C[choice_A][:,choice_B].shape)
        print('C_', C_.shape)
        sys.exit(0)
    # return torch.allclose(C, C_, rtol=rtol, atol=atol)
    return tclose
