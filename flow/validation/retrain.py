from flow.utils import vc, register_activation_hooks, register_gradient_hooks, TimeTracker, Logger, tensors_close

def validate_retrain(validation_set, model, optimizer, loss_fn, next_model, time_tracker: TimeTracker, logger: Logger, verbose=False, silent=False, index=None):

    time_tracker.start('validate_other')
    data, target, activations, gradients, loss = validation_set.get_dict().values()

    optimizer.zero_grad()

    verbose &= not silent

    # PREPARE MODEL FOR TRAINING STEP
    model.train()

    activations_check = register_activation_hooks(model)
    gradients_check = register_gradient_hooks(model)
    time_tracker.stop('validate_other')
    
    # TRAIN THE MODEL
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
            act_valid = tensors_close(act_check, act)
            act_total &= act_valid
            if verbose: print(f'    {vc(act_valid)} {key} (n: {val[0].shape[0]})')
    time_tracker.stop('validate_activations')
    
    # VALIDATE LOSS
    time_tracker.start('validate_loss')
    # loss_diff = abs(loss_check.item()-loss.item())/abs(loss_check.item())*100
    # loss_valid = loss_diff == 0.0
    loss_valid = tensors_close(loss_check, loss)
    if verbose: print('  LOSS:\n    {} DIFF[{}, {}]'.format(vc(loss_valid), loss_check.item(), loss.item()))
    time_tracker.stop('validate_loss')
    
    # VALIDATE GRADIENTS
    time_tracker.start('validate_gradients')
    if verbose: print('  GRADIENTS:')
    grad_total = True
    gradients_check = {key: getattr(model, key).weight.grad for key in activations.keys()}
    for key, grad_check in gradients_check.items():
        # grad_diff = torch.mean(torch.abs(grad_check - gradients[key][1]))*100
        # grad_valid = grad_diff == 0.0
        grad_valid = tensors_close(grad_check, gradients[key][1])
        grad_total &= grad_valid
        if verbose: print('    {} {}'.format(vc(grad_valid), key))
    time_tracker.stop('validate_gradients')
    
    # VALIDATE WEIGHTS
    time_tracker.start('validate_weights')
    if verbose: print('  WEIGHTS:')
    weight_total = True
    for (l, weight), next_weight in zip(model.named_parameters(), next_model.parameters()):
        # weight_diff = torch.sum(torch.abs(weight - next_weight))*100
        # weight_valid = weight_diff == 0.0
        weight_valid = tensors_close(weight, next_weight)
        weight_total &= weight_valid
        if verbose: print('    {} {}'.format(vc(weight_valid), l))
    time_tracker.stop('validate_weights')

    if not silent and not verbose: 
        if index is not None: print(f'batch {index:04d}', end=' ')
        print(f'act: {vc(act_total)}; loss: {vc(loss_valid)}; grad: {vc(grad_total)}; weights: {vc(weight_total)}')
    
    if verbose: print()