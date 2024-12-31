import torch
import torch.nn as nn
import torch.nn.functional as F
from pet_age_estimation.models.backbone import BackboneFeatureExtractor
from pet_age_estimation.models.multi_branch_head import MultiBranchRegressionHead

class CrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** 0.5
        
    def forward(self, x1, x2):
        # x1, x2: [batch_size, feature_dim]
        batch_size = x1.size(0)
        
        # Project queries, keys, and values
        Q = self.query(x1)  # [batch_size, feature_dim]
        K = self.key(x2)    # [batch_size, feature_dim]
        V = self.value(x2)  # [batch_size, feature_dim]
        
        # Reshape for attention computation
        Q = Q.view(batch_size, 1, -1)      # [batch_size, 1, feature_dim]
        K = K.view(batch_size, -1, 1)      # [batch_size, feature_dim, 1]
        V = V.view(batch_size, 1, -1)      # [batch_size, 1, feature_dim]
        
        # Compute attention scores
        attention = torch.bmm(Q, K) / self.scale  # [batch_size, 1, 1]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(attention, V)  # [batch_size, 1, feature_dim]
        out = out.squeeze(1)           # [batch_size, feature_dim]
        
        return out

class FeatureFusionModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        
        # Cross-attention for bidirectional feature interaction
        self.cross_attention1 = CrossAttention(feature_dim)
        self.cross_attention2 = CrossAttention(feature_dim)
        
        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, features1, features2):
        # Cross-attention interaction
        attended1 = self.cross_attention1(features1, features2)
        attended2 = self.cross_attention2(features2, features1)
        
        # Concatenate original and attended features
        combined = torch.cat([
            features1, features2, 
            attended1, attended2
        ], dim=1)
        
        # Generate gating weights
        gates = self.gate(combined)
        
        # Refine features
        refined = self.refinement(combined)
        
        # Apply gating
        fused = refined * gates
        
        return fused

class DualBranchAgeEstimator(nn.Module):
    def __init__(self,
                 num_branches=8,
                 dropout_rate=0.3):
        super().__init__()
        
        # Store input sizes for each branch
        self.vit_size = 336
        self.convnext_size = 384
        
        # Primary branch: ViT with adaptive pooling
        self.backbone1 = nn.Sequential(
            BackboneFeatureExtractor(
                backbone_type='vit',
                pretrained=True,
                freeze_backbone=True
            ),
            nn.AdaptiveAvgPool2d((1, 1))  # If needed
        )
        
        # Secondary branch: ConvNeXt with adaptive pooling
        self.backbone2 = nn.Sequential(
            BackboneFeatureExtractor(
                backbone_type='convnextv2_tiny',
                pretrained=True,
                freeze_backbone=True
            ),
            nn.AdaptiveAvgPool2d((1, 1))  # If needed
        )
        
        # Get feature dimensions
        feature_dim1 = self.backbone1[0].get_feature_dimensions()
        feature_dim2 = self.backbone2[0].get_feature_dimensions()
        
        # Feature adaptation layers to match dimensions
        self.adapt1 = nn.Linear(feature_dim1, 512)
        self.adapt2 = nn.Linear(feature_dim2, 512)
        
        # Feature fusion module
        self.fusion = FeatureFusionModule(512)
        
        # Multi-branch regression head
        self.regression_head = MultiBranchRegressionHead(
            input_dim=512,
            num_branches=num_branches
        )
        
        # Feature enhancement
        self.enhancement = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 512),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 512),
                nn.GELU()
            ) for _ in range(2)
        ])
    
    def unfreeze_backbones(self):
        """Unfreeze both backbones completely"""
        self.backbone1[0].unfreeze_backbone()
        self.backbone2[0].unfreeze_backbone()
    
    def freeze_backbones(self):
        """Freeze both backbones completely"""
        self.backbone1[0]._freeze_backbone()
        self.backbone2[0]._freeze_backbone()
    
    def partial_unfreeze(self, num_layers=4):
        """Unfreeze the last n layers of both backbones"""
        self.backbone1[0].partial_unfreeze(num_layers)
        self.backbone2[0].partial_unfreeze(num_layers)
        
    def forward(self, x):
        # Resize input for each branch
        x1 = F.interpolate(x, size=(self.vit_size, self.vit_size), 
                          mode='bilinear', align_corners=False)
        x2 = F.interpolate(x, size=(self.convnext_size, self.convnext_size), 
                          mode='bilinear', align_corners=False)
        
        # Extract features from both branches with appropriately sized inputs
        features1 = self.backbone1[0](x1)  # ViT branch
        features2 = self.backbone2[0](x2)  # ConvNeXt branch
        
        # Adapt feature dimensions
        features1 = self.adapt1(features1)
        features2 = self.adapt2(features2)
        
        # Enhance features independently
        enhanced1 = self.enhancement[0](features1)
        enhanced2 = self.enhancement[1](features2)
        
        # Fuse features
        fused_features = self.fusion(enhanced1, enhanced2)
        
        # Multi-branch regression
        prediction, branch_weights = self.regression_head(fused_features)
        
        # Get branch outputs for consistency loss
        branch_outputs = []
        for i in range(len(self.regression_head.branches)):
            branch_out = self.regression_head.branches[i](fused_features)
            branch_outputs.append(branch_out)
        branch_outputs = torch.stack(branch_outputs, dim=1)
        
        # Combine outputs
        output = {
            'prediction': prediction.squeeze(-1),
            'branch_routing_weights': branch_weights,
            'branch_outputs': branch_outputs  # For consistency loss
        }
        
        return output