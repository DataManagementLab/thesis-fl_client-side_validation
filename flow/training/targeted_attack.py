
from flow.utils import Logger

def targeted_attack(model, optimizer, loss_fn, data, target, epoch, batch, device, logger: Logger, boosting=False, boost_factor=10):
    target, malicious = target
    malicious = malicious.all().item()
    optimizer.zero_grad()
    if malicious and boosting:
        orig_lr = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] *= boost_factor
    output = model(data.to(device).view(-1, 28 * 28))
    loss = loss_fn(output, target.to(device))
    loss.backward()
    optimizer.step()
    if malicious and boosting: 
        optimizer.param_groups[0]['lr'] = orig_lr
        logger.log_attack_application(epoch, batch, 'all', 'GRADIENT_BOOSTING')
    return loss