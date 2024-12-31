import torch  
import torch.nn as nn  
import torch.nn.functional as F  

class ProbabilisticAgeLoss(nn.Module):  
    def __init__(self,   
                 mae_weight=1.0,      # MAE loss weight  
                 range_weight=0,      # Weight for range-aware loss  
                 consistency_weight=0):   # Weight for consistency loss  
        super().__init__()  
        
        self.mae_weight = mae_weight  
        self.range_weight = range_weight  
        self.consistency_weight = consistency_weight  
        
        # Base loss functions  
        self.mae_loss = nn.L1Loss(reduction='none')  
        self.mse_loss = nn.MSELoss(reduction='none')  
    
    def forward(self, outputs, targets):  
        """  
        Compute loss with essential components  
        
        Args:  
            outputs (dict): Model predictions dictionary containing:  
                - prediction (tensor): Main age predictions [batch_size, 1]  
                - branch_routing_weights (tensor): Branch attention weights [batch_size, num_branches]  
                - branch_outputs (tensor): Individual branch predictions [batch_size, num_branches, 1]  
            targets (tensor): True age labels [batch_size]  
        
        Returns:  
            tuple: (total_loss, loss_components)  
                - total_loss (tensor): Combined weighted loss  
                - loss_components (dict): Individual loss terms for logging  
        """  
        predictions = outputs['prediction']  
        branch_outputs = outputs['branch_outputs']  
        
        # 1. Base MAE Loss  
        mae_loss = self.mae_loss(predictions, targets).mean()  
        
        # 2. Range-Aware Loss  
        pred_range = torch.max(predictions) - torch.min(predictions)  
        target_range = torch.max(targets) - torch.min(targets)  
        range_loss = torch.abs(pred_range - target_range)  
        
        # 3. Consistency Loss between branches  
        branch_preds = branch_outputs.squeeze(-1)  # [batch_size, num_branches]  
        branch_mean = torch.mean(branch_preds, dim=1)  # [batch_size]  
        consistency_loss = torch.tensor(0., device=predictions.device)  
        for i in range(branch_preds.size(1)):  
            consistency_loss = consistency_loss + self.mse_loss(  
                branch_preds[:, i],  
                branch_mean  
            ).mean()  
        consistency_loss = consistency_loss / branch_preds.size(1)  
        
        # Combine essential losses  
        total_loss = (  
            self.mae_weight * mae_loss +  
            self.range_weight * range_loss +  
            self.consistency_weight * consistency_loss  
        )  
        
        # Log individual loss components  
        loss_components = {  
            'mae_loss': mae_loss.item(),  
            'range_loss': range_loss.item(),  
            'consistency_loss': consistency_loss.item()  
        }  
        
        return total_loss, loss_components  

# Additional Utility Loss Functions  
def symmetric_mean_absolute_percentage_error(preds, targets):  
    """  
    Symmetric Mean Absolute Percentage Error  
    More robust to outliers compared to standard MAPE  
    """  
    return torch.mean(  
        2 * torch.abs(preds - targets) /   
        (torch.abs(preds) + torch.abs(targets) + 1e-8)  
    ) * 100  

def huber_loss(preds, targets, delta=1.0):  
    """  
    Huber Loss: Less sensitive to outliers  
    """  
    error = torch.abs(preds - targets)  
    quadratic_mask = error <= delta  
    linear_mask = ~quadratic_mask  
    
    quadratic_loss = 0.5 * error[quadratic_mask] ** 2  
    linear_loss = delta * (error[linear_mask] - 0.5 * delta)  
    
    return torch.mean(torch.cat([quadratic_loss, linear_loss]))