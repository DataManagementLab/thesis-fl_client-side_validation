
def no_attack(model, optimizer, loss_fn, data, target):
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    return loss