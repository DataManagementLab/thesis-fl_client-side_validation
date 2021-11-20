import yaml, torch

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
    # return torch.all(torch.lt(torch.abs(torch.sub(tensor1, tensor2)), atol)).item()
    return torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
