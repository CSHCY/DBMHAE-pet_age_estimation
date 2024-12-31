import pytest
import torch
import torch.nn as nn
from pet_age_estimation.models.uncertainty_module import UncertaintyModule, ProbabilisticAgeEstimator

@pytest.fixture
def uncertainty_module():
    return UncertaintyModule(input_dim=512, dropout_rate=0.3)

@pytest.fixture
def probabilistic_estimator():
    return ProbabilisticAgeEstimator(backbone_type='convnextv2_huge', num_branches=8, dropout_rate=0.3)

class TestUncertaintyModule:
    def test_initialization(self, uncertainty_module):
        assert isinstance(uncertainty_module, nn.Module)
        assert isinstance(uncertainty_module.uncertainty_feature_extractor, nn.Sequential)
        assert isinstance(uncertainty_module.uncertainty_head, nn.Sequential)
        assert isinstance(uncertainty_module.epistemic_head, nn.Sequential)

    def test_forward_training(self, uncertainty_module):
        batch_size = 4
        input_features = torch.randn(batch_size, 512)
        branch_predictions = torch.randn(batch_size)
        
        output = uncertainty_module(input_features, branch_predictions, training=True)
        
        assert isinstance(output, dict)
        assert 'prediction' in output
        assert 'aleatoric_uncertainty' in output
        assert 'epistemic_uncertainty' in output
        assert 'aleatoric_uncertainty_confidence_score' in output
        
        assert output['prediction'].shape == (batch_size,)
        assert output['aleatoric_uncertainty'].shape == (batch_size,)
        assert output['epistemic_uncertainty'].shape == (batch_size,)
        assert output['aleatoric_uncertainty_confidence_score'].shape == (batch_size,)
        
        # Check that uncertainties are non-negative
        assert torch.all(output['aleatoric_uncertainty'] >= 0)
        assert torch.all(output['epistemic_uncertainty'] >= 0)
        
        # Check confidence scores are between 0 and 1
        assert torch.all(output['aleatoric_uncertainty_confidence_score'] >= 0)
        assert torch.all(output['aleatoric_uncertainty_confidence_score'] <= 1)

    def test_forward_inference(self, uncertainty_module):
        batch_size = 4
        input_features = torch.randn(batch_size, 512)
        branch_predictions = torch.randn(batch_size)
        
        output = uncertainty_module(input_features, branch_predictions, training=False)
        
        # During inference, epistemic uncertainty should be zeros
        assert torch.all(output['epistemic_uncertainty'] == 0)

class TestProbabilisticAgeEstimator:
    def test_initialization(self, probabilistic_estimator):
        assert isinstance(probabilistic_estimator, nn.Module)
        assert hasattr(probabilistic_estimator, 'backbone')
        assert hasattr(probabilistic_estimator, 'regression_head')
        assert hasattr(probabilistic_estimator, 'uncertainty_module')

    def test_forward_training(self, probabilistic_estimator):
        batch_size = 4
        channels = 3
        height = 224
        width = 224
        
        input_tensor = torch.randn(batch_size, channels, height, width)
        output = probabilistic_estimator(input_tensor, training=True)
        
        assert isinstance(output, dict)
        assert 'prediction' in output
        assert 'aleatoric_uncertainty' in output
        assert 'epistemic_uncertainty' in output
        assert 'branch_routing_weights' in output
        assert 'aleatoric_uncertainty_confidence_score' in output
        
        assert output['prediction'].shape == (batch_size,)
        assert output['aleatoric_uncertainty'].shape == (batch_size,)
        assert output['epistemic_uncertainty'].shape == (batch_size,)
        assert output['aleatoric_uncertainty_confidence_score'].shape == (batch_size,)
        
        # Check branch routing weights shape (batch_size x num_branches)
        assert output['branch_routing_weights'].shape == (batch_size, 8)
        
        # Check that branch routing weights sum to 1 for each sample
        assert torch.allclose(output['branch_routing_weights'].sum(dim=1), 
                            torch.ones(batch_size), 
                            atol=1e-6)

    def test_forward_inference(self, probabilistic_estimator):
        batch_size = 4
        channels = 3
        height = 224
        width = 224
        
        input_tensor = torch.randn(batch_size, channels, height, width)
        output = probabilistic_estimator(input_tensor, training=False)
        
        # During inference, epistemic uncertainty should be zeros
        assert torch.all(output['epistemic_uncertainty'] == 0)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_different_batch_sizes(self, probabilistic_estimator, batch_size):
        channels = 3
        height = 224
        width = 224
        
        input_tensor = torch.randn(batch_size, channels, height, width)
        output = probabilistic_estimator(input_tensor)
        
        assert output['prediction'].shape == (batch_size,)
        assert output['aleatoric_uncertainty'].shape == (batch_size,)
        assert output['branch_routing_weights'].shape == (batch_size, 8) 