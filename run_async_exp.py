from multiprocessing import Process
from flow.multiprocessing import training_process, get_process_logger, start_validators, stop_validators
 
 
if __name__ == '__main__':
     
    # Some lists with our favorite characters
    names = [['Master Shake', 'Meatwad', 'Frylock', 'Carl'],
             ['Early', 'Rusty', 'Sheriff', 'Granny', 'Lil'],
             ['Rick', 'Morty', 'Jerry', 'Summer', 'Beth']]

    consumers, queue, lock = start_validators(len(names) * 2)

    producers = []

    for n in names:
        # Create our producer processes by passing the producer function and it's arguments
        producers.append(Process(target=training_process, args=(queue, lock, n)))
 
    # Start the producers and consumer
    # The Python VM will launch new independent processes for each Process object
    for p in producers: p.start()

    logger = get_process_logger()
 
    # Like threading, we have a join() method that synchronizes our program
    for p in producers: p.join()
    stop_validators(consumers, queue)
 
    logger.info('Parent process exiting...')