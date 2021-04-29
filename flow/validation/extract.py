def validate_batch_freivald(data, target, activations, gradients, model, optimizer, loss, next_model, verbose=False, index=None):

    optimizer.zero_grad()

    # VALIDATE ACTIVATIONS
    save_input = dict()
    if verbose: print('  ACTIVATIONS:')
    act_total = True
    data = data.view(-1, 28 * 28)
    for key, val in activations.items():
        save_input[key] = torch.clone(data)
        layer = getattr(model, key)
        act_valid = freivald(data, layer.weight.T, val[0], bias=layer.bias, rtol=1e-05, atol=5e-06)
        # act_valid = baseline(data, layer.weight.T, val[0], bias=layer.bias, rtol=1e-05, atol=5e-06)
        if verbose: print(f'    {vc(act_valid)} {key} (n: {val[0].shape[0]})')
        act_total &= act_valid
        data = F.relu(val[0])
    
    # VALIDATE LOSS
    loss_input = torch.clone(data)
    loss_input.requires_grad = True
    loss_check = loss_fn(loss_input, target)
    loss_diff = abs(loss_check.item()-loss.item())/abs(loss_check.item())*100
    loss_valid = loss_diff == 0.0
    if verbose: print('  LOSS:\n    {} DIFF[{}, {}] = {}%'.format(vc(loss_valid), loss_check.item(), loss.item(), loss_diff))
    # data.retain_grad()
    loss_check.backward()
    
    # VALIDATE GRADIENT
    if verbose: print('  GRADIENTS:')
    grad_total = True
    C = loss_input.grad.clone()

    for key, val in reversed(list(gradients.items())) :

        layer = getattr(model, key)
        W = layer.weight.clone()
        b = layer.bias.clone()

        I = save_input[key]
        A = activations[key][0]
        C_a = (A > 0).float() * C

        grad_W = torch.mm(C_a.T, I)
        grad_b = torch.sum(C_a, dim=0)
        grad_x = torch.mm(C_a, W)

        grad_valid = freivald(C_a.T, I, val, atol=1e-06)
        # grad_valid = baseline(C_a.T, I, val, atol=1e-06)
        if verbose: print(f'    {vc(grad_valid)} {key} (n: {val.shape[0]})')

        layer.weight.grad = val
        layer.bias.grad = grad_b

        C = grad_x
        grad_total &= grad_valid
        
        # next_layer = getattr(next_model, key)
        # W_new = W + va

    # VALIDATE NEW WEIGHTS
    if verbose: print('  WEIGHTS:')
    optimizer.step()
    weight_total = True

    for key in gradients.keys():
        new_layer = getattr(model, key)
        next_layer = layer = getattr(next_model, key)
        W_valid = torch.allclose(new_layer.weight, next_layer.weight)
        b_valid = torch.allclose(new_layer.bias, next_layer.bias)
        if verbose: print(f'    {key} {vc(W_valid)} (weight) {vc(b_valid)} (bias)')
        weight_total &= W_valid & b_valid

    # PRINT RESULT SUMMARY
    if False and not verbose: 
        if index: pass
        print(f'batch {index:04d}', end=' ')
        print(f'act: {vc(act_total)}; loss: {vc(loss_valid)}; grad: {vc(grad_total)}; weight: {vc(weight_total)}')