

def validate_batch_baseline(data, target, activations, gradients, model, optimizer, loss, next_model, verbose=False, index=None):

    # PREPARE MODEL FOR TRAINING STEP
    model.train()
    hooks_check, activations_check = register_activation_hooks(model)
    
    # TRAIN THE MODEL
    optimizer.zero_grad()
    output = model(data)
    loss_check = loss_fn(output, target)
    loss_check.backward()
    optimizer.step()
    
    size_print = 16
    
    # VALIDATE ACTIVATIONS
    if verbose: print('  ACTIVATIONS:')
    act_total = True
    for key, val in activations_check.items():
        if verbose: print(f'    {key} (n: {val[0].shape[0]})')
        for act_check, act in zip(val, activations[key]):
            act_diff = (torch.mean(torch.abs(act_check - act), dim=1)*100).tolist()
            act_print = [vc(ad == 0.0) for ad in act_diff]
            for ad in act_diff: act_total &= ad == 0.0
            len_print = math.ceil(len(act_print) / size_print)
            if verbose:
                for i in range(len_print):
                    print('      ' + ' '.join(act_print[size_print*(i):size_print*(i+1)]))
    
    # VALIDATE LOSS
    loss_diff = abs(loss_check.item()-loss.item())/abs(loss_check.item())*100
    loss_valid = loss_diff == 0.0
    if verbose: print('  LOSS:\n    {} DIFF[{}, {}] = {}%'.format(vc(loss_valid), loss_check.item(), loss.item(), loss_diff))
    
    # VALIDATE GRADIENT
    if verbose: print('  GRADIENTS:')
    grad_total = True
    gradients_check = {key: getattr(model, key).weight.grad for key in activations.keys()}
    for key, grad_check in gradients_check.items():
        grad_diff = torch.mean(torch.abs(grad_check - gradients[key]))*100
        grad_valid = grad_diff == 0.0
        grad_total &= grad_valid.item()
        if verbose: print('    {} {} [{}%]'.format(vc(grad_valid), key, grad_diff))
    
    # VALIDATE WEIGHTS
    if verbose: print('  WEIGHTS:')
    weight_total = True
    for (l, weight), next_weight in zip(model.named_parameters(), next_model.parameters()):
        weight_diff = torch.sum(torch.abs(weight - next_weight))*100
        weight_valid = weight_diff == 0.0
        weight_total &= weight_valid.item()
        if verbose: print('    {} {}'.format(vc(weight_valid), l))

    # if not verbose: 
    #     if index: print(f'batch {index:04d}', end=' ')
    #     print(f'act: {vc(act_total)}; loss: {vc(loss_valid)}; grad: {vc(grad_total)}; weights: {vc(weight_total)}')
    
    if verbose: print()