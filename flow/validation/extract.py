import torch
import torch.nn.functional as F

from flow.utils import TimeTracker, Logger, ValidationSet, vc, tensors_close, rand_true

def validate_extract(validation_method, validation_set: ValidationSet, model, optimizer, loss_fn, next_model, time_tracker: TimeTracker, logger: Logger, val_prob=None, verbose=False, silent=False, index=None, **method_args):

    # logger.log_attack_detection(validation_set.epoch, dict(msg="epoch: {}, batch: {}".format(*validation_set.get_id())))

    data, target, activations, gradients, loss = validation_set.get_dict().values()

    optimizer.zero_grad()

    verbose &= not silent

    # VALIDATE ACTIVATIONS
    time_tracker.start('validate_activations')
    save_input = dict()
    if verbose: print('  ACTIVATIONS:')
    act_total = True
    data = data.view(-1, 28 * 28)
    for i, (key, val) in enumerate(activations.items()):
        save_input[key] = torch.clone(data)
        if rand_true(val_prob):
            layer = getattr(model, key)
            act_valid = validation_method(data, layer.weight.T, val[0], bias=layer.bias, rtol=1e-05, atol=1e-04, **method_args)
            if not act_valid:
                logger.log_attack_detection(
                    validation_set.epoch, 
                    validation_set.batch,
                    key,
                    'ACTIVATION'
                )
            if verbose: print(f'    {vc(act_valid)} {key} (n: {val[0].shape[0]})')
            act_total &= act_valid
        if False and i+1 == len(activations):
            data = F.softmax(val[0], dim=1)
        else:
            data = F.relu(val[0])
    time_tracker.stop('validate_activations')
    
    # VALIDATE LOSS
    time_tracker.start('validate_loss')
    loss_input = torch.clone(data)
    loss_input.requires_grad = True
    loss_check = loss_fn(loss_input, target)
    # loss_diff = abs(loss_check.item()-loss.item())/abs(loss_check.item())*100
    # loss_valid = loss_diff == 0.0
    loss_valid = tensors_close(loss_check, loss)
    if not loss_valid:
        logger.log_attack_detection(
            validation_set.epoch, 
            validation_set.batch,
            key,
            'LOSS'
        )
    if verbose: print('  LOSS:\n    {} DIFF[{}, {}]'.format(vc(loss_valid), loss_check.item(), loss.item()))
    # data.retain_grad()
    loss_check.backward()
    time_tracker.stop('validate_loss')
    
    # VALIDATE GRADIENTS
    time_tracker.start('validate_gradients')
    if verbose: print('  GRADIENTS:')
    grad_total = True
    C = loss_input.grad.clone()

    for key, val in list(gradients.items()) :

        grad_x, grad_W, grad_b = val
        layer = getattr(model, key)

        if rand_true(val_prob):

            W = layer.weight.clone()
            b = layer.bias.clone()

            I = save_input[key]
            A = activations[key][0]
            C_a = torch.mul(torch.gt(A, 0).float(), C)

            if grad_x is not None:
                grad_x_valid = validation_method(C_a, W, grad_x, atol=1e-05)
            else:
                grad_x_valid = True
            grad_W_valid = validation_method(C_a.T, I, grad_W, atol=1e-05)
            # grad_b_valid = torch.allclose(torch.sum(C_a, dim=0), grad_b, atol=1e-06)
            grad_b_valid = tensors_close(torch.sum(C_a, dim=0), grad_b)

            # if not grad_x_valid: print(f'Detected Epoch: {validation_set.epoch}, Batch: {validation_set.batch}, Weight: {key}.input')
            # if not grad_W_valid: print(f'Detected Epoch: {validation_set.epoch}, Batch: {validation_set.batch}, Weight: {key}.weight')
            # if not grad_b_valid: print(f'Detected Epoch: {validation_set.epoch}, Batch: {validation_set.batch}, Weight: {key}.bias')

            if not grad_x_valid: 
                logger.log_attack_detection(
                    validation_set.epoch, 
                    validation_set.batch,
                    key + '.input',
                    'GRADIENT'
                )
            if not grad_W_valid: 
                logger.log_attack_detection(
                    validation_set.epoch, 
                    validation_set.batch,
                    key + '.weight',
                    'GRADIENT'
                )
            if not grad_b_valid: 
                logger.log_attack_detection(
                    validation_set.epoch, 
                    validation_set.batch,
                    key + '.bias',
                    'GRADIENT'
                )

            grad_valid = grad_x_valid and grad_W_valid and grad_b_valid
            grad_total &= grad_valid
            
            if verbose: print(f'    {vc(grad_valid)} {key} (n: {val[1].shape[0]})')

        layer.weight.grad = grad_W
        layer.bias.grad = grad_b

        C = grad_x
        
    time_tracker.stop('validate_gradients')

    # VALIDATE NEW WEIGHTS
    time_tracker.start('validate_weights')
    if verbose: print('  WEIGHTS:')
    time_tracker.start('validate_weights_optimizer')
    optimizer.step()
    time_tracker.stop('validate_weights_optimizer')
    weight_total = True

    for key in gradients.keys():
        if rand_true(val_prob):
            time_tracker.start('validate_weights_getattr')
            new_layer = getattr(model, key)
            next_layer = getattr(next_model, key)
            time_tracker.stop('validate_weights_getattr')
            time_tracker.start('validate_weights_allclose')
            # W_valid = torch.allclose(new_layer.weight, next_layer.weight)
            # b_valid = torch.allclose(new_layer.bias, next_layer.bias)
            W_valid = tensors_close(new_layer.weight, next_layer.weight)
            b_valid = tensors_close(new_layer.bias, next_layer.bias)
            time_tracker.stop('validate_weights_allclose')

            if not W_valid: 
                logger.log_attack_detection(
                    validation_set.epoch, 
                    validation_set.batch,
                    key,
                    'WEIGHT_B'
                )
            if not b_valid: 
                logger.log_attack_detection(
                    validation_set.epoch, 
                    validation_set.batch,
                    key,
                    'WEIGHT_B'
                )
            if verbose: print(f'    {key} {vc(W_valid)} (weight) {vc(b_valid)} (bias)')
            weight_total &= W_valid & b_valid
    time_tracker.stop('validate_weights')

    # for (key, new_layer), next_layer in zip(model.named_parameters(), next_model.parameters()):
    #     if rand_true(val_prob):
    #         time_tracker.start('validate_weights_allclose')
    #         param_valid = tensors_close(new_layer, next_layer)
    #         time_tracker.stop('validate_weights_allclose')
    #         if verbose: print(f'    {key} {vc(param_valid)}')
    #         weight_total &= param_valid
    # time_tracker.stop('validate_weights')

    # PRINT RESULT SUMMARY
    if not silent and not verbose: 
        if index: pass
        print(f'batch {index:04d}', end=' ')
        print(f'act: {vc(act_total)}; loss: {vc(loss_valid)}; grad: {vc(grad_total)}; weight: {vc(weight_total)}')