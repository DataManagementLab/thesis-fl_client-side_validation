from flow.utils import Logger

def no_attack(model, optimizer, loss_fn, data, target, epoch, batch, logger: Logger):
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    return loss