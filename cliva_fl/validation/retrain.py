from torch import nn
from copy import deepcopy
from cliva_fl.utils import vc, register_activation_hooks, register_gradient_hooks, TimeTracker, Logger, tensors_close

def validate_retrain(validation_set, model, optimizer, loss_fn, next_model, time_tracker: TimeTracker, logger: Logger, verbose=False, silent=False, index=None, rtol=1e-5, atol=1e-4):

    time_tracker.start('validate_other')
    data, target, activations, gradients, loss = validation_set.get_dict().values()

    model = deepcopy(model)
    next_model = deepcopy(next_model)

    optimizer.zero_grad()

    verbose &= not silent

    # PREPARE MODEL FOR TRAINING STEP
    model.train()

    activations_check = register_activation_hooks(model)
    gradients_check = register_gradient_hooks(model)
    time_tracker.stop('validate_other')
    
    # TRAIN THE MODEL
    # data = data.view(-1, 28 * 28)
    time_tracker.start('validate_retrain')
    optimizer.zero_grad()
    output = model(data)
    loss_check = loss_fn(output, target)
    loss_check.backward()
    optimizer.step()
    time_tracker.stop('validate_retrain')
    
    size_print = 16
    
    # VALIDATE ACTIVATIONS
    time_tracker.start('validate_activations')
    if verbose: print('  ACTIVATIONS:')
    act_total = True
    for key, val in activations_check.items():
        if verbose: print(f'    {key} (n: {val[0].shape[0]})')
        for act_check, act in zip(val, activations[key]):
            # act_diff = (torch.mean(torch.abs(act_check - act), dim=1)*100).tolist()
            # act_print = [vc(ad == 0.0) for ad in act_diff]
            # for ad in act_diff: act_total &= ad == 0.0
            # len_print = math.ceil(len(act_print) / size_print)
            act_valid = tensors_close(act_check, act, rtol=rtol, atol=atol)
            act_total &= act_valid
            if verbose: print(f'    {vc(act_valid)} {key} (n: {val[0].shape[0]})')
    time_tracker.stop('validate_activations')
    
    # VALIDATE LOSS
    time_tracker.start('validate_loss')
    # loss_diff = abs(loss_check.item()-loss.item())/abs(loss_check.item())*100
    # loss_valid = loss_diff == 0.0
    loss_valid = tensors_close(loss_check, loss, rtol=rtol, atol=atol)
    if verbose: print('  LOSS:\n    {} DIFF[{}, {}]'.format(vc(loss_valid), loss_check.item(), loss.item()))
    time_tracker.stop('validate_loss')
    
    # VALIDATE GRADIENTS
    time_tracker.start('validate_gradients')
    if verbose: print('  GRADIENTS:')
    grad_total = True
    # gradients_check = {key: getattr(model, key).weight.grad for key in activations.keys()}
    # for key, grad_check in gradients_check.items():
    for i in reversed(range(len(model.layers))):
        name = f'layers.{i}'
        module = model.layers[i]

        if type(module) in [nn.Linear, nn.Conv2d]:

            # grad_diff = torch.mean(torch.abs(grad_check - gradients[key][1]))*100
            # grad_valid = grad_diff == 0.0
            if name in gradients_check:
                grad_x_valid = tensors_close(gradients_check[name][0], gradients[name][0], rtol=rtol, atol=atol)
            else:
                grad_x_valid = True
            grad_W_valid = tensors_close(module.weight.grad, gradients[name][1], rtol=rtol, atol=atol)
            # grad_b_valid = torch.allclose(torch.sum(C_a, dim=0), grad_b, atol=1e-06)
            grad_b_valid = tensors_close(module.bias.grad, gradients[name][2], rtol=rtol, atol=atol)
            grad_valid = grad_x_valid and grad_W_valid and grad_b_valid
            grad_total &= grad_valid
            if verbose: print(f'    {vc(grad_valid)} {name} (n: {A.shape[0]})')
    time_tracker.stop('validate_gradients')
    
    # VALIDATE WEIGHTS
    time_tracker.start('validate_weights')
    if verbose: print('  WEIGHTS:')
    weight_total = True
    for (l, weight), next_weight in zip(model.named_parameters(), next_model.parameters()):
        # weight_diff = torch.sum(torch.abs(weight - next_weight))*100
        # weight_valid = weight_diff == 0.0
        weight_valid = tensors_close(weight, next_weight, rtol=rtol, atol=atol)
        weight_total &= weight_valid
        if verbose: print('    {} {}'.format(vc(weight_valid), l))
    time_tracker.stop('validate_weights')

    if not silent and not verbose: 
        if index is not None: print(f'batch {index:04d}', end=' ')
        print(f'act: {vc(act_total)}; loss: {vc(loss_valid)}; grad: {vc(grad_total)}; weights: {vc(weight_total)}')
    
    activations_check.clear()
    gradients_check.clear()

    if verbose: print()