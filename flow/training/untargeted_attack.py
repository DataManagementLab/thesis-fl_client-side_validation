import torch
from flow.utils import Logger
from flow.utils.model_poisoning import gradient_noise

def untargeted_attack(model, optimizer, loss_fn, data, target, epoch, batch, device, logger: Logger, frequency=0.1, scale=1/5):
    assert 0 < frequency < 1
    assert 0 < scale
    optimizer.zero_grad()
    output = model(data.to(device))
    loss = loss_fn(output, target.to(device))
    loss.backward()
    for l, weight in model.named_parameters():
        if torch.rand(1).item() <= frequency:
            noise = gradient_noise(weight.shape, scale=scale, device=device)
            weight.grad += noise
            logger.log_attack_application(epoch, batch, l, "GRADIENT_NOISE")
            # print(f'\tWeight: {l}')
    optimizer.step()
    return loss