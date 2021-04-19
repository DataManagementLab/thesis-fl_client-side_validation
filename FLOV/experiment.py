from utils import ValidationSet

class Experiment:
    def __init__(model, optimizer, loss, dataset, validation_method):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dataset = dataset
        self.validation_method = validation_method

        self.buffer = dict()

    def run(n_epochs, max_buffer_len=100):
        model.train()

        train_start_time = time.time()
        for epoch in range(n_epochs):
            # monitor training loss
            train_loss = 0.0

            train_batch_start = time.time()
            
            for index, (data, target) in enumerate(train_loader):
                # SAVE TO SET
                vset = ValidationSet(epoch, batch, self.validation_method)
                vset.set_data(data, target)
                vset.set_model_start(model)
                vset.set_optimizer(optimizer)
                
                # TRAINING
                hooks, activations = register_activation_hooks(model)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                
                gradients = {key: getattr(model, key).weight.grad for key in activations.keys()}

                # SAVE TO SET
                vset.set_loss(loss)
                vset.set_activations(activations)
                vset.set_gradients(gradients)
                vset.set_model_end(model)

                buffer[index] = vset

                activations = {key: list() for key in activations.keys()}
                
                train_loss += loss.item()*data.size(0)

                if len(buffer) >= max_buffer_len:
                    train_batch_end = time.time()
                    
                    gc.collect()
                    
                    validate_batch_start = time.time()
                    validate_buffer(buffer)
                    validate_batch_end = time.time()

                    info(f"Time\t{(train_batch_end - train_batch_start):.4f} training\t{(validate_batch_end - validate_batch_start):.4f} validation")

                    buffer = dict()
                    gc.collect()
                    train_batch_start = time.time()
                    break

            if len(buffer) >= max_buffer_len:
                validate_buffer(buffer)
                buffer = dict()
            
            train_loss = train_loss/len(train_loader.dataset)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))

        train_end_time = time.time()

        print(f'Execution time: {(train_end_time - train_start_time):.4f} sec')

    def validate(buffer):
    for index, (data, target, model_state_dict, next_model_state_dict, optimizer_state_dict, loss, activations, gradients) in buffer.items():
        model = M(layers)
        next_model = M(layers)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        model.load_state_dict(model_state_dict)
        next_model.load_state_dict(next_model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

        validate_batch(
            method=self.validation_method,
            data=data, 
            target=target, 
            activations=activations, 
            gradients=gradients, 
            model=model, 
            optimizer=optimizer, 
            loss=loss, 
            next_model=next_model,
            index=index,
            verbose=False)
        # break
    
    def validate_batch(data, target, activations, gradients, model, optimizer, loss, next_model, verbose=False, index=None):
        pass

    def stats():
        pass

    def reset():
        self.buffer = dict()
        gc.collect()