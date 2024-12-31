import torch  
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau  
from torch.cuda.amp import autocast, GradScaler  

class TrainingConfig:  
    def __init__(self,   
                 learning_rate=1e-4,  
                 weight_decay=1e-5,  
                 gradient_clip=1.0,  
                 mixed_precision=True,  
                 fine_tuning_lr_factor=0.3,  # Learning rate factor for fine-tuning  
                 partial_fine_tuning_lr_factor=0.8):  # Learning rate factor for partial fine-tuning  
        self.learning_rate = learning_rate  
        self.weight_decay = weight_decay  
        self.gradient_clip = gradient_clip  
        self.mixed_precision = mixed_precision  
        self.fine_tuning_lr_factor = fine_tuning_lr_factor  
        self.partial_fine_tuning_lr_factor = partial_fine_tuning_lr_factor  

class TrainingManager:  
    def __init__(self, model, config):  
        self.model = model  
        self.config = config  
        
        # Optimizer  
        self.optimizer = torch.optim.AdamW(  
            model.parameters(),   
            lr=config.learning_rate,  
            weight_decay=config.weight_decay  
        )  
        
        # Learning Rate Scheduler  
        self.scheduler = CosineAnnealingLR(  
            self.optimizer,   
            T_max=50,  # Total epochs  
            eta_min=1e-6  
        )  
        
        # Mixed Precision  
        self.scaler = torch.amp.GradScaler(enabled=config.mixed_precision)  
    
    def train_step(self, batch, criterion):  
        """  
        Single training step with mixed precision support  
        """  
        self.model.train()  
        images, targets = batch  
        
        # Mixed Precision Training  
        with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):  
            outputs = self.model(images)  
            loss = criterion(outputs, targets)  
        
        # Gradient Scaling  
        self.scaler.scale(loss).backward()  
        
        # Gradient Clipping  
        self.scaler.unscale_(self.optimizer)  
        torch.nn.utils.clip_grad_norm_(  
            self.model.parameters(),   
            self.config.gradient_clip  
        )  
        
        # Optimizer Step  
        self.scaler.step(self.optimizer)  
        self.scaler.update()  
        
        return loss.item()