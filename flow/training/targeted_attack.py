
from flow.utils import Logger

def targeted_attack(model, optimizer, loss_fn, data, target, epoch, batch, device, logger: Logger, boosting=False, boost_factor=10):
    target, malicious = target
    malicious = malicious.all().item()
    optimizer.zero_grad()
    if malicious and boosting:
        # print('BOOSTING')
        orig_lr = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] *= boost_factor
        if False: print(f"lr_orig: {orig_lr}\tlr_now: {optimizer.param_groups[0]['lr']}")
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    if malicious and boosting: optimizer.param_groups[0]['lr'] = orig_lr
    return loss