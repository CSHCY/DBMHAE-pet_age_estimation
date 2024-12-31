import torch
import pytest
from ..models.multi_branch_head import MultiBranchRegressionHead

@pytest.fixture
def model_params():
    return {
        'input_dim': 512,
        'num_branches': 8,
        'batch_size': 4
    }

@pytest.fixture
def model(model_params):
    return MultiBranchRegressionHead(
        input_dim=model_params['input_dim'],
        num_branches=model_params['num_branches']
    )

@pytest.fixture
def sample_features(model_params):
    return torch.randn(model_params['batch_size'], model_params['input_dim'])

def test_initialization(model, model_params):
    """Test if the model initializes correctly"""
    assert len(model.branches) == model_params['num_branches']
    assert len(model.branch_ranges) == model_params['num_branches']

def test_forward_pass_shape(model, sample_features, model_params):
    """Test if the forward pass returns correct shapes"""
    predictions, routing_weights = model(sample_features)
    
    assert predictions.shape == (model_params['batch_size'], 1)
    assert routing_weights.shape == (model_params['batch_size'], model_params['num_branches'])

def test_routing_weights_sum(model, sample_features):
    """Test if routing weights sum to 1"""
    _, routing_weights = model(sample_features)
    sums = routing_weights.sum(dim=1)
    
    assert torch.allclose(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-5)

def test_custom_age_ranges(model_params):
    """Test if custom age ranges are properly set"""
    custom_ranges = [(0, 12), (13, 24), (25, 36), (37, 48)]
    model = MultiBranchRegressionHead(
        input_dim=model_params['input_dim'],
        num_branches=4,
        age_ranges=custom_ranges
    )
    assert model.branch_ranges == custom_ranges

def test_branch_outputs(model, sample_features, model_params):
    """Test if each branch produces valid outputs"""
    for branch in model.branches:
        output = branch(sample_features)
        assert output.shape == (model_params['batch_size'], 1)
        assert torch.all(torch.isfinite(output))

@pytest.mark.parametrize("input_dim,num_branches", [
    (256, 4),
    (512, 8),
    (1024, 6)
])
def test_different_configurations(input_dim, num_branches):
    """Test model with different input dimensions and branch counts"""
    model = MultiBranchRegressionHead(input_dim=input_dim, num_branches=num_branches)
    features = torch.randn(2, input_dim)
    predictions, routing_weights = model(features)
    
    assert predictions.shape == (2, 1)
    assert routing_weights.shape == (2, num_branches)

def test_invalid_input_shape(model):
    """Test if model raises error with incorrect input shape"""
    invalid_features = torch.randn(4, 256)  # Wrong input dimension
    with pytest.raises(RuntimeError):
        model(invalid_features) 