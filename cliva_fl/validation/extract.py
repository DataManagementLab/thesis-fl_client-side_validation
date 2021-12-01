import torch
from torch import nn
import torch.nn.functional as F

from cliva_fl.utils import TimeTracker, Logger, ValidationSet, vc, tensors_close, rand_true

def validate_extract(validation_method, validation_set: ValidationSet, model, optimizer, loss_fn, next_model, time_tracker: TimeTracker, logger: Logger, val_prob=None, verbose=False, silent=False, rtol=1e-5, atol=1e-4, index=None, **method_args):

    data, target, activations, gradients, loss = validation_set.get_dict().values()

    optimizer.zero_grad()

    verbose &= not silent

    # VALIDATE ACTIVATIONS
    time_tracker.start('validate_activations')
    save_input = dict()
    if verbose: print('  ACTIVATIONS:')
    act_total = True

    for name, module in model.named_modules():
        if type(module) in [nn.Linear, nn.Conv2d]:
            save_input[name] = data.detach().clone()
            output = activations[name][0]
            if rand_true(val_prob):
                act_valid = validation_method(data, module.weight.T, output, bias=module.bias, rtol=rtol, atol=atol, **method_args)
                if not act_valid:
                    logger.log_attack_detection(
                        validation_set.epoch, 
                        validation_set.batch,
                        name,
                        'ACTIVATION'
                    )
                if verbose: print(f'    {vc(act_valid)} {name} (n: {output.shape[0]})')
                act_total &= act_valid
            data = F.relu(output)
    time_tracker.stop('validate_activations')
    
    # VALIDATE LOSS
    time_tracker.start('validate_loss')
    data.requires_grad = True
    loss_check = loss_fn(data, target)
    loss_valid = tensors_close(loss_check, loss, rtol=rtol, atol=atol)
    if not loss_valid:
        logger.log_attack_detection(
            validation_set.epoch, 
            validation_set.batch,
            'loss',
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
    C = data.grad.clone()

    for i in reversed(range(len(model.layers))):

        name = f'layers.{i}'
        module = model.layers[i]

        if type(module) in [nn.Linear, nn.Conv2d]:

            grad_x, grad_W, grad_b = gradients[name]

            if rand_true(val_prob):

                W = module.weight.clone()
                b = module.bias.clone()

                I = save_input[name]
                A = activations[name][0]
                C_a = torch.mul(torch.gt(A, 0).float(), C)

                if grad_x is not None:
                    grad_x_valid = validation_method(C_a, W, grad_x, rtol=rtol, atol=atol, **method_args)
                else:
                    grad_x_valid = True
                grad_W_valid = validation_method(C_a.T, I, grad_W, rtol=rtol, atol=atol, **method_args)
                grad_b_valid = tensors_close(torch.sum(C_a, dim=0), grad_b, rtol=rtol, atol=atol)

                if not grad_x_valid: 
                    logger.log_attack_detection(
                        validation_set.epoch, 
                        validation_set.batch,
                        name + '.input',
                        'GRADIENT'
                    )
                if not grad_W_valid: 
                    logger.log_attack_detection(
                        validation_set.epoch, 
                        validation_set.batch,
                        name + '.weight',
                        'GRADIENT'
                    )
                if not grad_b_valid: 
                    logger.log_attack_detection(
                        validation_set.epoch, 
                        validation_set.batch,
                        name + '.bias',
                        'GRADIENT'
                    )

                grad_valid = grad_x_valid and grad_W_valid and grad_b_valid
                grad_total &= grad_valid
                
                if verbose: print(f'    {vc(grad_valid)} {name} (n: {A.shape[0]})')

            module.weight.grad = grad_W
            module.bias.grad = grad_b

            C = grad_x
        
    time_tracker.stop('validate_gradients')

    # VALIDATE NEW WEIGHTS
    time_tracker.start('validate_weights')
    if verbose: print('  WEIGHTS:')
    time_tracker.start('validate_weights_optimizer')
    optimizer.step()
    time_tracker.stop('validate_weights_optimizer')
    weight_total = True

    for i in range(len(model.layers)):

        name = f'layers.{i}'
        module = model.layers[i]

        if type(module) in [nn.Linear, nn.Conv2d]:

            if rand_true(val_prob):
                time_tracker.start('validate_weights_getattr')
                next_module = next_model.layers[i]
                time_tracker.stop('validate_weights_getattr')
                time_tracker.start('validate_weights_allclose')
                W_valid = tensors_close(module.weight, next_module.weight, rtol=rtol, atol=atol)
                b_valid = tensors_close(module.bias, next_module.bias, rtol=rtol, atol=atol)
                time_tracker.stop('validate_weights_allclose')

                if not W_valid: 
                    logger.log_attack_detection(
                        validation_set.epoch, 
                        validation_set.batch,
                        name,
                        'WEIGHT_W'
                    )
                if not b_valid: 
                    logger.log_attack_detection(
                        validation_set.epoch, 
                        validation_set.batch,
                        name,
                        'WEIGHT_B'
                    )
                if verbose: print(f'    {name} {vc(W_valid)} (weight) {vc(b_valid)} (bias)')
                weight_total &= W_valid & b_valid
    time_tracker.stop('validate_weights')

    # PRINT RESULT SUMMARY
    if not silent and not verbose: 
        if index: pass
        print(f'batch {index:04d}', end=' ')
        print(f'act: {vc(act_total)}; loss: {vc(loss_valid)}; grad: {vc(grad_total)}; weight: {vc(weight_total)}')
    