import torch
from backbone import BackboneFeatureExtractor

def print_feature_dimensions():
    # Define backbone types and their required input sizes
    backbone_configs = [
        ('convnextv2_tiny', 384),
        ('convnextv2_huge', 384),
        ('vit', 336),
        ('convnextv2_tiny_feature_maps', 384)
    ]
    
    for backbone_type, input_size in backbone_configs:
        print(f"\nTesting {backbone_type} (input size: {input_size}x{input_size}):")
        # Create input tensor with appropriate size
        x = torch.randn(2, 3, input_size, input_size)
        
        model = BackboneFeatureExtractor(backbone_type=backbone_type, pretrained=False)
        model.eval()
        
        with torch.no_grad():
            features = model(x)
            
        if isinstance(features, list):
            print("Feature maps dimensions:")
            for i, feat in enumerate(features):
                print(f"Level {i}: {feat.shape}")
        else:
            print(f"Feature dimension: {features.shape}")
            print(f"Feature dimension: {model.get_feature_dimensions()}")

if __name__ == "__main__":
    print_feature_dimensions() 