import copy, time, gc
from cliva_fl.utils import ValidationBuffer

def validate_buffer(buffer: ValidationBuffer, validation_fn, model, optimizer, loss_fn, time_tracker, logger, validation_delay=0):

    time_tracker.start('total_time_validation')

    next_model = copy.deepcopy(model)
    init_model_state = buffer.get_init_model_state()

    for index, vset in buffer.items():

        model.load_state_dict(init_model_state)
        next_model.load_state_dict(vset.get_model_state())
        optimizer.load_state_dict(vset.get_optimizer_state())

        time_tracker.start('raw_time_validation')
        validation_fn(
            index=index,
            validation_set=vset,
            model=model, 
            optimizer=optimizer,
            next_model=next_model,
            loss_fn=loss_fn,
            time_tracker=time_tracker,
            logger=logger
        )
        time_tracker.stop('raw_time_validation')
        
        init_model_state = vset.get_model_state()

    del next_model
    gc.collect()
    
    time_tracker.stop('total_time_validation')
    time.sleep(validation_delay * time_tracker.last('total_time_validation'))
