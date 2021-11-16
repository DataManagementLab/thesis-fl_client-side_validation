import torch
# from memory_profiler import profile
# @profile
def validate_buffer(buffer, validation_fn, model_builder, optimizer_builder, loss_fn_builder, time_tracker, logger):

    time_tracker.start('total_time_validation')

    model = model_builder()
    next_model = model_builder()
    optimizer = optimizer_builder(model.parameters())
    loss_fn = loss_fn_builder()
    device = torch.device("cpu")
    
    for index, vset in buffer.items():

        model.load_state_dict(vset.get_model_start())
        model.to(device)
        next_model.load_state_dict(vset.get_model_end())
        next_model.to(device)
        optimizer.load_state_dict(vset.get_optimizer())

        time_tracker.start('raw_time_validation')
        validation_fn(
            model=model, 
            optimizer=optimizer, 
            loss_fn=loss_fn, 
            next_model=next_model,
            time_tracker=time_tracker,
            logger=logger,
            index=index,
            validation_set=vset
        )
        time_tracker.stop('raw_time_validation')
    
    time_tracker.stop('total_time_validation')
