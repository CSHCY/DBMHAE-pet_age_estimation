import torch.nn as nn  
import timm  

class BackboneFeatureExtractor(nn.Module):  
    def __init__(self,   
                 backbone_type='convnextv2_huge',   
                 pretrained=True,   
                 num_classes=0,
                 freeze_backbone=True):  
        super().__init__()  
        
        backbone_image_size_configs = [
        ('convnextv2_tiny', 384),
        ('convnextv2_huge', 384),
        ('vit', 336),
        ('convnextv2_tiny_feature_maps', 384)]
        # Select backbone  
        if backbone_type == 'convnextv2_tiny':  
            self.backbone = timm.create_model(  
                'convnextv2_tiny.fcmae_ft_in22k_in1k_384',   
                pretrained=pretrained,   
                num_classes=num_classes,  
            ) 
        elif backbone_type == 'convnextv2_tiny_feature_maps':
            self.backbone = timm.create_model(
                'convnextv2_tiny.fcmae_ft_in22k_in1k_384',
                pretrained=pretrained,
                num_classes=num_classes,
                features_only=True
            )
        elif backbone_type == 'convnextv2_huge':
            self.backbone = timm.create_model(
                'convnextv2_huge.fcmae_ft_in22k_in1k_384',
                pretrained=pretrained,
                num_classes=num_classes,
            )
        elif backbone_type == 'vit':  
            self.backbone = timm.create_model(  
                'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k',   
                pretrained=pretrained,   
                num_classes=num_classes,  
            )
        else:  
            raise ValueError(f"Unsupported backbone: {backbone_type}")  
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
            
    def _freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def partial_unfreeze(self, num_layers=2):
        """Unfreeze only the last few layers"""
        # First freeze everything
        self._freeze_backbone()
        
        # Then unfreeze the last layers
        for name, param in reversed(list(self.backbone.named_parameters())):
            if 'head' in name or 'norm' in name:  # Adjust based on model architecture
                param.requires_grad = True
                num_layers -= 1
                if num_layers <= 0:
                    break
    
    def get_feature_dimensions(self):
        return self.backbone.feature_info[-1]['num_chs']
    
    def forward(self, x):  
        # Extract multi-scale features  
        features = self.backbone(x)  
        return features  
