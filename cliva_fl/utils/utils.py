import yaml, torch
from math import log, pow, ceil

def vc(res): 
    return "\U00002705" if res else "\U0000274C"

def load_config(config_file: str, *element_list: list):
    
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    if len(element_list) == 0: 
        return config
    else: 
        return (config[el] for el in element_list)

def rand_true(prob: float = None) -> bool:
    if prob is None: return True
    assert 0 <= prob < 1
    return torch.rand(1).item() <= prob

def tensors_close(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol=1e-07, atol=1e-06) -> bool:
    # ∣input−other∣ <= atol + rtol x ∣other∣
    return torch.lt(torch.abs(torch.sub(tensor1, tensor2)), atol + rtol * torch.abs(tensor2)).all().item()
    # ∣input−other∣ <= atol
    # return torch.all(torch.lt(torch.abs(torch.sub(tensor1, tensor2)), atol)).item()
    # torch.allclose
    # return torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)

def tensors_close_sum(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol=1e-07, atol=1e-06) -> bool:
    res = True
    for i in range(len(tensor1.shape)):
        res &= tensors_close(tensor1.sum(i), tensor2.sum(i), rtol, atol)
    return res

def freivalds_rounds(n_layers, guarantee):
    return ceil(log(1 - pow(guarantee, 1/(3*n_layers-1)),0.5))
