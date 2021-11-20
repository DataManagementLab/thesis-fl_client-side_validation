from copy import deepcopy

class ValidationSet():

    def __init__(self, epoch, batch, is_initial: bool = False):
        self.set_id(epoch, batch)
        self.set_initial(is_initial)
        
        # Predefine Attributes
        self.data = self.target = self.model_state_dict = self.optimizer_state_dict = self.activations = self.gradients = self.loss = None
    
    def set_initial(self, initial: bool):
        self.it_initial = initial
    
    def set_id(self, epoch: int, batch: int):
        self.epoch = epoch
        self.batch = batch
    
    def set_data(self, data, target):
        self.data = data.detach().cpu()
        self.target = target.detach().cpu()
    
    def set_model_state(self, model):
        self.model_state_dict = { k: v.detach().clone().cpu() for k, v in model.state_dict().items() }
    
    def set_optimizer_state(self, optimizer):
        self.optimizer_state_dict = deepcopy(optimizer.state_dict())
    
    def set_activations(self, activations):
        self.activations = deepcopy(activations)
    
    def set_gradients(self, gradients):
        self.gradients = deepcopy(gradients)
    
    def set_loss(self, loss):
        self.loss = loss.detach().cpu()
    
    def get_id(self):
        return self.epoch, self.batch
    
    def get_data(self):
        return self.data, self.target
    
    def get_model_state(self):
        return self.model_state_dict
    
    def get_optimizer_state(self):
        return self.optimizer_state_dict 
    
    def get_activations(self):
        return self.activations
    
    def get_gradients(self):
        return self.gradients
    
    def get_loss(self):
        return self.loss
    
    def get_dict(self):
        return dict(
            data=self.data,
            target=self.target,
            activations=self.activations,
            gradients=self.gradients,
            loss=self.loss
        )

    @property
    def is_complete(self):
        complete = True
        complete &= self.data is not None
        complete &= self.target is not None
        complete &= self.model_start_state_dict is not None
        complete &= self.model_end_state_dict is not None
        complete &= self.optimizer_state_dict is not None
        complete &= self.activations is not None
        complete &= self.gradients is not None
        complete &= self.loss is not None
        return complete
    