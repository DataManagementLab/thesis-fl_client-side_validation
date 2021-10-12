
def validate_buffer(buffer, validation_fn, model_builder, optimizer_builder, loss_fn_builder, time_tracker, logger):

    time_tracker.start('total_time_validation')

    model = model_builder()
    next_model = model_builder()
    optimizer = optimizer_builder(model.parameters())
    loss_fn = loss_fn_builder()
    
    for index, vset in buffer.items():

        model.load_state_dict(vset.get_model_start())
        next_model.load_state_dict(vset.get_model_end())
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
