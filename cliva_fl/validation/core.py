import tracemalloc, copy, sys, time, gc
from torch.profiler import profile, schedule

from cliva_fl.utils import ValidationBuffer

def validate_buffer(buffer: ValidationBuffer, validation_fn, model, optimizer, loss_fn, time_tracker, logger, validation_delay=0):

    time_tracker.start('total_time_validation')
    # t1 = time.time()

    next_model = copy.deepcopy(model)
    init_model_state = buffer.get_init_model_state()
    
    # tracemalloc.start()
    # with profile(
    #     schedule=schedule(
    #         wait=2,
    #         warmup=2,
    #         active=6),
    #         # repeat=1),
    #     profile_memory=True
    #     # on_trace_ready=tensorboard_trace_handler,
    #     # with_stack=True
    # ) as profiler:

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
        # profiler.step()
    
    # print(profiler.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))
    # print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"{current:0.2f}, {peak:0.2f}")
    # tracemalloc.stop()
    # t2 = time.time()
    # time.sleep(2*(t2-t1))

    del next_model
    gc.collect()
    # sys.exit(0)
    
    time_tracker.stop('total_time_validation')
    time.sleep(validation_delay * time_tracker.last('total_time_validation'))
    # sys.exit(0)
