import torch
from flow.utils.model_poisoning import gradient_noise

def untargeted_attack(model, optimizer, loss_fn, data, target, frequency=0.1, scale=1/5):
    assert 0 < frequency < 1
    assert 0 < scale
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    for l, weight in model.named_parameters():
        if torch.rand(1).item() <= frequency:
            noise = gradient_noise(weight.shape, scale=scale)
            # print(f'\tWeight: {l}')
            weight.grad += noise
    optimizer.step()
    return loss