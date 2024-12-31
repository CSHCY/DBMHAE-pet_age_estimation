import pytest
import torch
from pet_age_estimation.models.backbone import BackboneFeatureExtractor


@pytest.fixture
def sample_input_384():
    # Create a sample batch of 2 images with 3 channels and 384x384 resolution
    return torch.randn(4, 3, 384, 384)

@pytest.fixture
def sample_input_336():
    # Create a sample batch of 2 images with 3 channels and 336x336 resolution
    return torch.randn(4, 3, 336, 336)

def test_convnextv2_tiny_initialization():
    model = BackboneFeatureExtractor(backbone_type='convnextv2_tiny', pretrained=True)
    assert isinstance(model, BackboneFeatureExtractor)

def test_convnextv2_huge_initialization():
    model = BackboneFeatureExtractor(backbone_type='convnextv2_huge', pretrained=True)
    assert isinstance(model, BackboneFeatureExtractor)

def test_vit_initialization():
    model = BackboneFeatureExtractor(backbone_type='vit', pretrained=True)
    assert isinstance(model, BackboneFeatureExtractor)

def test_invalid_backbone():
    with pytest.raises(ValueError):
        BackboneFeatureExtractor(backbone_type='invalid_backbone')

def test_convnextv2_tiny_forward(sample_input_384):
    model = BackboneFeatureExtractor(backbone_type='convnextv2_tiny', pretrained=True)
    output = model(sample_input_384)
    assert isinstance(output, torch.Tensor)
    assert output.dim() == 2  # [batch_size, embedding_dim]
    assert output.size(0) == sample_input_384.size(0)  # Batch size should match
    print(output.size())
    
def test_convnextv2_tiny_feature_maps_forward(sample_input_384):
    model = BackboneFeatureExtractor(
        backbone_type='convnextv2_tiny_feature_maps', 
        pretrained=True
    )
    outputs = model(sample_input_384)
    
    assert isinstance(outputs, list)
    assert len(outputs) == 4  # ConvNeXt typically outputs 4 feature maps
    
    for feat_map in outputs:
        assert isinstance(feat_map, torch.Tensor)
        assert feat_map.dim() == 4
        assert feat_map.size(0) == sample_input_384.size(0)  # Batch size should match

def test_convnextv2_huge_forward(sample_input_384):
    model = BackboneFeatureExtractor(backbone_type='convnextv2_huge', pretrained=True)
    output = model(sample_input_384)
    assert isinstance(output, torch.Tensor)
    assert output.dim() == 2
    print(output.size())
    assert output.size(0) == sample_input_384.size(0)

def test_vit_forward(sample_input_336):
    model = BackboneFeatureExtractor(backbone_type='vit', pretrained=True)
    output = model(sample_input_336)
    assert isinstance(output, torch.Tensor)
    assert output.dim() == 2
    assert output.size(0) == sample_input_336.size(0)
    print(output.size())
    
def test_vit_wrong_input_size(sample_input_384):
    model = BackboneFeatureExtractor(backbone_type='vit', pretrained=True)
    # Should raise an error when using 384x384 input with ViT
    with pytest.raises(Exception):
        _ = model(sample_input_384)
