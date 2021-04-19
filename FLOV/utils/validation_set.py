
class ValidationSet():

    def __init__(self, epoch, batch, validation_method):
        self.epoch = epoch
        self.batch = batch
        self.validation_method = validation_method
        
        # Predefine Attributes
        self.data = self.target = self.model_start_state_dict = self.model_end_state_dict = self.optimizer_state_dict = self.activations = self.gradients = self.loss = None
    
    def set_data(self, data, target):
        self.data = data
        self.target = target
    
    def set_model_start(self, model_start):
        self.model_start_state_dict = deepcopy(model_start.state_dict())
    
    def set_model_end(self, model_end):
        self.model_end_state_dict = deepcopy(model_end.state_dict())
    
    def set_optimizer(self, optimizer):
        self.optimizer_state_dict = deepcopy(optimizer.state_dict())
    
    def set_activations(self, activations):
        self.activations = deepcopy(activations)
    
    def set_gradients(self, gradients):
        self.gradients = deepcopy(gradients)
    
    def set_loss(self, loss):
        self.loss = loss
    
    def get_id(self):
        return self.epoch, self.batch
    
    def get_validation_method(self):
        return self.validation_method
    
    def get_data(self):
        return self.data, self.target
    
    def get_model_start(self):
        return self.model_start_state_dict
    
    def get_model_end(self):
        return self.model_end_state_dict
    
    def get_optimizer(self):
        return self.optimizer_state_dict 
    
    def get_activations(self):
        return self.activations
    
    def get_gradients(self):
        return self.gradients
    
    def get_loss(self):
        return self.loss

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