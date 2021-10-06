from multiprocessing import Process, Queue, Lock
from flow.multiprocessing import training_process, validation_process
 
 
if __name__ == '__main__':
     
    # Some lists with our favorite characters
    names = [['Master Shake', 'Meatwad', 'Frylock', 'Carl'],
             ['Early', 'Rusty', 'Sheriff', 'Granny', 'Lil'],
             ['Rick', 'Morty', 'Jerry', 'Summer', 'Beth']]
 
    # Create the Queue object
    queue = Queue()
     
    # Create a lock object to synchronize resource access
    lock = Lock()
 
    producers = []
    consumers = []
 
    for n in names:
        # Create our producer processes by passing the producer function and it's arguments
        producers.append(Process(target=training_process, args=(queue, lock, n)))
 
    for i in range(len(names) * 2):
        # Create our consumer processes by passing the consumer function and it's arguments
        consumers.append(Process(target=validation_process, args=(queue, lock)))
 
    # Start the producers and consumer
    # The Python VM will launch new independent processes for each Process object
    for p in producers: p.start()
    for c in consumers: c.start()
 
    # Like threading, we have a join() method that synchronizes our program
    for p in producers: p.join()
    for c in consumers: queue.put(None)
    for c in consumers: c.join()
 
    print('Parent process exiting...')