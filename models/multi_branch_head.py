import torch  
import torch.nn as nn  

class MultiBranchRegressionHead(nn.Module):  
    def __init__(self, input_dim, num_branches=8):  
        super().__init__()  
        
        # Branch-specific regression heads  
        self.branches = nn.ModuleList([  
            nn.Sequential(  
                nn.Linear(input_dim, input_dim // 2),  
                nn.GELU(),  
                nn.Dropout(0.2),  
                nn.Linear(input_dim // 2, 1)  
            ) for _ in range(num_branches)  
        ])  
        
        # Attention mechanism for branch weighting  
        self.attention = nn.Sequential(  
            nn.Linear(input_dim, input_dim // 2),  
            nn.GELU(),  
            nn.Dropout(0.2),  
            nn.Linear(input_dim // 2, num_branches),  
            nn.Softmax(dim=1)  
        )  
    
    def forward(self, x):  
        # Get attention weights for branches  
        branch_weights = self.attention(x)  # [batch_size, num_branches]  
        
        # Get predictions from each branch  
        branch_predictions = []  
        for branch in self.branches:  
            pred = branch(x)  # [batch_size, 1]  
            branch_predictions.append(pred)  
        
        # Stack branch predictions  
        branch_predictions = torch.stack(branch_predictions, dim=1)  # [batch_size, num_branches, 1]  
        
        # Weighted sum of branch predictions  
        weighted_prediction = torch.sum(  
            branch_predictions.squeeze(-1) * branch_weights,   
            dim=1,  
            keepdim=True  
        )  # [batch_size, 1]  
        
        return weighted_prediction, branch_weights