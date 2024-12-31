import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from pet_age_estimation.models.backbone import BackboneFeatureExtractor  
from pet_age_estimation.models.multi_branch_head import MultiBranchRegressionHead  

class UncertaintyModule(nn.Module):  
    def __init__(self, input_dim, dropout_rate=0.3):  
        super().__init__()  
        
        # Shared feature extraction for uncertainty  
        self.uncertainty_feature_extractor = nn.Sequential(  
            nn.Linear(input_dim, 256),  
            nn.GELU(),  
            nn.Dropout(dropout_rate)  
        )  
        
        # Uncertainty estimation head  
        # Now focuses purely on uncertainty quantification  
        self.uncertainty_head = nn.Sequential(  
            nn.Linear(256, 128),  
            nn.GELU(),  
            nn.Dropout(dropout_rate),  
            nn.Linear(128, 2)  # Variance parameters for the branch predictions  
        )  
        
        # Epistemic uncertainty head  
        self.epistemic_head = nn.Sequential(  
            nn.Linear(256, 128),  
            nn.GELU(),  
            nn.Dropout(dropout_rate),  
            nn.Linear(128, 1)  # Epistemic uncertainty estimation  
        )  
    
    def forward(self, features, branch_predictions, training=True):  
        # Extract uncertainty features  
        uncertainty_features = self.uncertainty_feature_extractor(features)  
        
        # Estimate uncertainty parameters for branch predictions  
        uncertainty_params = self.uncertainty_head(uncertainty_features)  
        log_var = uncertainty_params[:, 1]  # variance
        confidence = uncertainty_params[:, 0]
        
        # Epistemic uncertainty estimation  
        if training:  
            # Monte Carlo Dropout for epistemic uncertainty  
            epistemic_outputs = torch.stack([  
                self.epistemic_head(uncertainty_features)   
                for _ in range(10)  # 10 forward passes  
            ], dim=1)  
            epistemic_uncertainty = torch.std(epistemic_outputs, dim=1).squeeze(-1)  
        else:  
            epistemic_uncertainty = torch.zeros_like(branch_predictions)  
        
        # Variance is softplus transformed to ensure positivity  
        variance = F.softplus(log_var)  
        
        return {  
            'prediction': branch_predictions,  # Use branch predictions directly  
            'aleatoric_uncertainty': variance,  
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty_confidence_score': torch.sigmoid(confidence)
        }  

class ProbabilisticAgeEstimator(nn.Module):  
    def __init__(self,   
                 backbone = BackboneFeatureExtractor(), 
                 num_branches=8,  
                 dropout_rate=0.3):  
        super().__init__()  
        
        # Backbone Feature Extractor  
        self.backbone = backbone
        
        # Feature dimension from backbone  
        feature_dims = self.backbone.get_feature_dimensions()
        
        # Multi-Branch Regression Head  
        self.regression_head = MultiBranchRegressionHead(  
            input_dim=feature_dims,  
            num_branches=num_branches  
        )  
        
        # Uncertainty Module  
        self.uncertainty_module = UncertaintyModule(  
            input_dim=feature_dims,  
            dropout_rate=dropout_rate  
        )  
    
    def forward(self, x, training=True):  
        # Extract features  
        features = self.backbone(x)  
        
        global_features = features  # Use full features since backbone already returns last layer  
        
        # Multi-branch regression  
        branch_prediction, routing_weights = self.regression_head(global_features)  
        
        # Uncertainty estimation  
        uncertainty_output = self.uncertainty_module(  
            global_features,   
            branch_prediction.squeeze(-1),  # Pass branch predictions  
            training=training  
        )  
        
        # Combine outputs  
        output = {  
            'prediction': uncertainty_output['prediction'],  
            'aleatoric_uncertainty': uncertainty_output['aleatoric_uncertainty'], 
            'aleatoric_uncertainty_confidence_score': uncertainty_output['aleatoric_uncertainty_confidence_score'],
            'epistemic_uncertainty': uncertainty_output['epistemic_uncertainty'],  
            'branch_routing_weights': routing_weights  
        }  
        
        return output