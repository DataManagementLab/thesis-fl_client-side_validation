from flow.utils import Logger

def no_attack(model, optimizer, loss_fn, data, target, epoch, batch, device, logger: Logger):
    optimizer.zero_grad()
    output = model(data.to(device).view(-1, 28 * 28))
    loss = loss_fn(output, target.to(device))
    loss.backward()
    optimizer.step()
    return loss