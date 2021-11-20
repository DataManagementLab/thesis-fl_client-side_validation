
from cliva_fl.utils import ValidationSet

class ValidationBuffer:

    def __init__(self, epoch, buffer_size):
        self.epoch = epoch
        self.buffer_size = buffer_size
        self.buffer = dict()
    
    def set_init_model_state(self, model):
        self.init_model_state_dict = { k: v.detach().clone().cpu() for k, v in model.state_dict().items() }
    
    def get_init_model_state(self):
        return self.init_model_state_dict
    
    def add(self, batch: int, vset: ValidationSet):
        assert not self.full(), 'Buffer is full and more items can not be added.'
        self.buffer[batch] = vset
    
    def get(self, batch: int):
        return self.buffer[batch]
    
    def keys(self):
        return self.buffer.keys()
    
    def values(self):
        return self.buffer.values()
    
    def items(self):
        return self.buffer.items()
    
    def full(self):
        return self.size() >= self.buffer_size
    
    def size(self):
        return len(self.buffer)
    
    def clear(self):
        del self.init_model_state_dict
        self.buffer.clear()
