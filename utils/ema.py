import torch
import copy

class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        """
        Exponential Moving Average of model weights
        Args:
            model: model to apply EMA
            decay: EMA decay rate (should be close to 1)
            device: device to store EMA model
        """
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        
        if device is not None:
            self.ema.to(device)
            
        self.ema_parameters = dict(self.ema.named_parameters())
        self.model_parameters = dict(model.named_parameters())
        
        # Initialize EMA weights to model weights
        for name in self.ema_parameters:
            self.ema_parameters[name].data.copy_(self.model_parameters[name].data)
    
    def update(self, model):
        """
        Update EMA weights
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.ema_parameters:
                    self.ema_parameters[name].data.mul_(self.decay)
                    self.ema_parameters[name].data.add_(
                        param.data * (1 - self.decay)
                    )
    
    def state_dict(self):
        """Get EMA state dict"""
        return self.ema.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict"""
        self.ema.load_state_dict(state_dict) 