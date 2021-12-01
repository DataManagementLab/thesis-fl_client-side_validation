import torch, sys
from cliva_fl.utils import Logger
from cliva_fl.utils.model_poisoning import gradient_noise, rand_item

def untargeted_attack(model, optimizer, loss_fn, data, target, epoch, batch, device, logger: Logger, frequency=0.1, scale=1/5, random_noise=True, corrupt_neurons=None):
    assert 0 < frequency < 1 and 0 < scale
    optimizer.zero_grad()
    output = model(data.to(device))
    loss = loss_fn(output, target.to(device))
    loss.backward()
    for l, weight in model.named_parameters():
        if torch.rand(1).item() <= frequency:
            if corrupt_neurons:
                noise = torch.zeros(weight.shape)
                for n in range(corrupt_neurons):
                    noise[rand_item(noise.shape)] = gradient_noise((1,), scale=scale, device=device) if random_noise else scale
            else:
                noise = gradient_noise(weight.shape, scale=scale, device=device) if random_noise else torch.full(weight.shape, scale)
            weight.grad += noise
            logger.log_attack_application(epoch, batch, l, "GRADIENT")
    optimizer.step()
    return loss